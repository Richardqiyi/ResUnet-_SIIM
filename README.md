# ResUnet-_SIIM
ResUnet++ model for SIIM-ACR-Pneumothorax-Seg-XR dataset. For original repo, see [ResUnet++](https://github.com/rishikksh20/ResUnet/)

#### Environment

python = 3.7, PyTorch = 1.7.1, cuda = 10.1

#### GPU

1*Nvidia RTX 3090 24GB

#### Docker
```
docker pull stevezeyuzhang/colab:1.7.1
```

#### Installation

```
pip install -r requirements.txt
```

#### File Directory
```
|-- ResUnet-_SIIM
    |-- SIIM-ACR-Pneumothorax-Seg-XR (your dataset name)
        |-- trian
            |-- input (image folder)
                |-- <your image>
            |-- ouput (label folder)
                |-- <your label>(the same name with image)
        |-- valid
            |-- input (image folder)
                |-- <your image>
            |-- ouput (label folder)
                |-- <your label>(the same name with image)

    |-- SIIM_test (your test dataset)
            |-- input (image folder)
                |-- <your image>
```
#### Preprocess
You should resize the original images. Take `downsample.py` as an example. You can also use your own method.
```
python downsample.py
```
#### Training
The default hyperparameters are: optimizer: Adam; batch size = 8; shape= (512,512); epoch = 200,
You can download checkpoint [here](https://github.com/Richardqiyi/ResUnet-_SIIM/releases/tag/ResUnet%2B%2B_adam_bs8_shape512_epoch200_cosdecay)

```
python train.py
```
#### Inference

```
python inference.py
```
