import numpy as np
from torch.utils.data import Dataset

class MOSIDataset(Dataset):
    def __init__(self, path='data/MOSI/'):
        
        self.text = np.load(path + 'text.npy', allow_pickle=True)
        self.audio = np.load(path + 'audio.npy', allow_pickle=True) 
        self.video = np.load(path + 'video.npy', allow_pickle=True)
        self.labels = np.load(path + 'labels.npy')
        
    def preprocess(self):
     
        text_features = []
        for text in self.text:
            
            if len(text) > 50:
                text = text[:50]
            else:
                text = np.pad(text, (0, 50-len(text)))
            text_features.append(text)
            
        
        audio_features = []
        for audio in self.audio:
            
            mfcc = librosa.feature.mfcc(audio, sr=16000)
            if mfcc.shape[1] > 400:
                mfcc = mfcc[:,:400]
            else:
                mfcc = np.pad(mfcc, ((0,0),(0,400-mfcc.shape[1])))
            audio_features.append(mfcc)
            
        # 视频预处理
        video_features = []
        for video in self.video:
            # 提取关键帧特征,统一维度
            frames = video_to_frames(video)
            if len(frames) > 50:
                frames = frames[:50]
            else:
                frames = np.pad(frames, ((0,50-len(frames)),(0,0)))
            video_features.append(frames)
            
        return np.array(text_features), np.array(audio_features), np.array(video_features)

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'audio': self.audio[idx],
            'video': self.video[idx],
            'label': self.labels[idx]
        }


train_dataset = MOSIDataset('train')
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
