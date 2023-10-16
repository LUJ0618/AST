import os
import csv 
import torch
import torchaudio
import numpy as np
from torch.cuda.amp import autocast
import wget

from models import ASTModel

os.environ['TORCH_HOME'] = '../pretrained_models'

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
    if os.path.exists('../pretrained_models/audio_mdl.pth') is False:
        wget.download(audioset_mdl_url, out='../pretrained_models/audio_mdl.pth')

    # Assume each input spectrogram has 1024 time frames
    input_tdim = 1024
    checkpoint_path = '../pretrained_models/audio_mdl.pth'
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
    label_csv = '../egs/audioset/data/class_labels_indices.csv'
    labels = load_label(label_csv)

    # load sample files
    dir_path = '../sample_audios'
    file_names = os.listdir(dir_path)
    data_list = []
    
    # sample_audio_path = 'https://www.dropbox.com/s/vddohcnb9ane9ag/LDoXsip0BEQ_000177.flac?dl=1'
    # wget.download(sample_audio_path, '../sample_audios/sample_audio.flac')
    # file_name = '../sample_audios/sample_audio.flac'

    # feats = make_features(file_name, mel_bins=128)
    # feats_data = feats.expand(1, input_tdim, 128)
    # feats_data = feats_data.to(torch.device("cuda:0"))

    for file in file_names:

        file_path = os.path.join(dir_path, file)
        feats = make_features(file_path, mel_bins=128)
        feats_data = feats.expand(1, input_tdim, 128)
        feats_data = feats_data.to(torch.device("cuda:0"))  

        # Make the prediction
        with torch.no_grad():
            with autocast():
                output = ast_mdl.forward(feats_data)
                output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]

        
        
        data_list.append([file[:-4], result_output[0], result_output[513]])

        # target idx
        # 0 : Speech, 1 : Male Speech, 2 : Female Speech, 513 : Noise
        # target_idx = [0, 513]
        # Print audio tagging top probabilities
        print('Predice results:')
        # print(file[:-4])
        for k in range(10):
            print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))

    #import matplotlib
    #matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    names = [data[0] for data in data_list]
    # speech_vals = [data[1] for data in data_list]
    # mean_speech_vals = np.mean(speech_vals)

    noise_vals = [data[2] for data in data_list]
    mean_noise_vals = np.mean(noise_vals)

    
    plt.scatter(names, noise_vals, label="데이터 포인트")
    plt.xlabel(file_names)
    # plt.ylabel(f"Speech Avg : {mean_speech_vals}")
    plt.ylabel(f"Noise Avg : {mean_noise_vals}")
    plt.title("Clean Data Noise probs")

    # for _, data in enumerate(data_list):
    #     plt.annotate(data[0], (data[1], data[2]))

    plt.legend()
    plt.grid(True)
    plt.savefig("Clean Data Noise probs")
    #plt.show()



if __name__ == "__main__":
    main()

