# ResUnet-_SIIM
ResUnet++ model for SIIM-ACR-Pneumothorax-Seg-XR dataset. For original repo, see [ResUnet++](https://github.com/rishikksh20/ResUnet/)

#### Environment

python = 3.7, PyTorch = 1.7.1, cuda = 10.1

#### Docker

stevezeyuzhang/colab:1.7.1

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


