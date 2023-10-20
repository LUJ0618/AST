import os
import csv 
import torch
import torchaudio
import numpy as np
from torch.cuda.amp import autocast
import wget

from src.models import ASTModel

os.environ['TORCH_HOME'] = './pretrained_models'

class ASTModelVis(ASTModel):
    def get_att_map(self, block, x):
        qkv = block.attn.qkv
        num_heads = block.attn.num_heads
        scale = block.attn.scale
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

def make_features(wav_name, mel_bins, target_length=1024):
    waveform, sr = torchaudio.load(wav_name)
    assert sr == 16000, 'input audio sampling rate must be 16kHz'

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels

def main():

    # Create an AST model and download the AudioSet pretrained weights
    audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
    if os.path.exists('./pretrained_models/audio_mdl.pth') is False:
        wget.download(audioset_mdl_url, out='./pretrained_models/audio_mdl.pth')

    # Assume each input spectrogram has 1024 time frames
    input_tdim = 1024
    checkpoint_path = './pretrained_models/audio_mdl.pth'
    # now load the visualization model
    ast_mdl = ASTModelVis(
        label_dim=527,
        input_tdim=input_tdim,
        imagenet_pretrain=True,
        audioset_pretrain=True
    )
    

    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()        

    # Load the AudioSet label set
    label_csv = './egs/audioset/data/class_labels_indices.csv'
    labels = load_label(label_csv)

    #############################################################################
    # 두 폴더를 나타내는 디렉토리 경로를 정의합니다.
    dir_path_clean = './samples_clean'  # 첫 번째 폴더의 경로로 교체합니다.
    dir_path_noisy = './samples_noisy'  # 두 번째 폴더의 경로로 교체합니다.

    # 각 폴더의 파일 목록을 가져옵니다.
    # file_names_clean = os.listdir(dir_path_clean)
    # file_names_noisy = os.listdir(dir_path_noisy)

    # 빨간색 그룹과 파란색 그룹의 데이터를 저장할 리스트를 만듭니다.
    data_list_clean = []
    data_list_noisy = []

    # 각 그룹에 대한 색상을 지정하는 리스트를 만듭니다.
    color_red = 'red'
    color_blue = 'blue'

    # 폴더를 처리하고 해당 그룹에 데이터를 추가하는 함수를 정의합니다.
    def process_folder(folder_path, data_list, color):
        
        # tags_dict = {}
        # file_tags_list = []
        del_tag_list = [0, 7, 5, 2, 4, 1, 506, 525, 31, 500,
                        26, 504, 491, 137, 507, 378, 524, 46, 72, 44, 
                        407, 510, 459, 453, 387, 417, 360, 408, 55, 48,
                        73, 442, 24, 448, 41, 3, 81, 83]

        for file in os.listdir(folder_path):
            # print(file[:-4])
            
            file_path = os.path.join(folder_path, file)
            feats = make_features(file_path, mel_bins=128)
            feats_data = feats.expand(1, input_tdim, 128)
            feats_data = feats_data.to(torch.device("cuda:0"))

            sum_top10_probs = 0

            with torch.no_grad():
                with autocast():
                    output = ast_mdl.forward(feats_data)
                    output = torch.sigmoid(output)

            result_output = output.data.cpu().numpy()[0]
            for i in del_tag_list:
                result_output[i] = 0

            # del result_output[0]
            sorted_indexes = np.argsort(result_output)[::-1]
            # sum_top10_probs += sum(result_output[sorted_indexes[:10]])

            for k in range(10):
                # file_tags_list.append(np.array(labels)[sorted_indexes[k]])
                
                print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
                sum_top10_probs = sum_top10_probs + result_output[sorted_indexes[k]]
                print(sum_top10_probs)
            
        # for tag in file_tags_list:
        #     if tag in tags_dict:
        #         tags_dict[tag] += 1
        #     else:
        #         tags_dict[tag] = 1

        # return tags_dict


            # 해당 데이터를 해당 그룹의 데이터 리스트에 추가합니다.
            data_list.append([file[:-4], sum_top10_probs, result_output[513], color])

    # '빨간색' 폴더를 처리하고 빨간색 그룹에 데이터를 추가합니다.
    process_folder(dir_path_clean, data_list_clean, color_red)
    # filtered_element_count = {element: count for element, count in clean_tags_dict.items() if count >= 15}
    
    # for tag, count in filtered_element_count.items():
    #     print(f"tag: {tag}, count: {count}")

    

    # '파란색' 폴더를 처리하고 파란색 그룹에 데이터를 추가합니다.
    process_folder(dir_path_noisy, data_list_noisy, color_blue)
    # filtered_element_count = {element: count for element, count in noisy_tags_dict.items() if count >= 15}
    # for tag, count in filtered_element_count.items():
    #     print(f"tag: {tag}, count: {count}")

    threshold = 0.1
    max_value_clean = max(item[1] for item in data_list_clean)
    min_value_clean = min(item[1] for item in data_list_clean)

    max_value_noisy = max(item[1] for item in data_list_noisy)
    min_value_noisy = min(item[1] for item in data_list_noisy)

    max_value = max(max_value_clean, max_value_noisy) + 0.1
    min_value = min(min_value_clean, min_value_noisy)

    current_value = min_value

    while current_value <= max_value:
        filtered_data_clean = [data for data in data_list_clean if data[1] < current_value]
        print(len(filtered_data_clean)/50)
        filtered_data_noisy = [data for data in data_list_noisy if data[1] < current_value]
        print((50-len(filtered_data_noisy))/50)
        print(current_value)

        print('#####################')

        current_value += 0.1

    filtered_data_clean = [data for data in data_list_clean if data[1] < threshold]
    print(len(filtered_data_clean))
    print(len(filtered_data_clean)/50)
    filtered_data_noisy = [data for data in data_list_noisy if data[1] < threshold]
    print(len(filtered_data_noisy))
    print((50-len(filtered_data_noisy))/50)

    import matplotlib.pyplot as plt

    # 빨간색 그룹과 파란색 그룹의 데이터 포인트를 다른 색상과 레이블로 플롯합니다.
    for data in data_list_clean:
        plt.scatter(data[0], data[1], label=data[0], color=data[3])

    for data in data_list_noisy:
        plt.scatter(data[0], data[1], label=data[0], color=data[3])

    # 그래프 설정
    plt.axhline(y=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('File_names')
    plt.ylabel('Top10_Probs_Sum')
    plt.title('Clean vs Noisy Data Top10 Probs Sum')
    # plt.legend(loc='upper right')
    plt.savefig("Clean vs Noisy Data Top10 Probs Sum")
    plt.show()



    #############################################################################




if __name__ == "__main__":
    main()

