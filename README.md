# vaik-segmentation-trt-experiment

Create json file by segmentation model. Calc mIoU.

![draw](https://user-images.githubusercontent.com/116471878/222900493-a2606e90-58c7-427e-9ae2-7d05b3e74e6b.png)


## Install

```shell
pip install -r requirements.txt
```

## Docker Install

- amd64(g4dn.xlarge)

```shell
docker build -t g4dnxl_ed_experiment -f ./Dockerfile.g4dn.xlarge .
sudo docker run --runtime=nvidia \
           --name g4dnxl_ed_experiment_container \
           --rm \
           -v ~/.vaik_segmentation_pb_trainer/dump_dataset:/workspace/dump_dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it g4dnxl_ed_experiment /bin/bash
```

- arm64(JetsonXavierNX)

```shell
sudo docker build -t jxnj502_experiment -f ./Dockerfile.jetson_xavier_nx_jp_502 .
sudo docker run --runtime=nvidia \
           --name jxnj502_experiment_container \
           --rm \
           -v ~/.vaik_segmentation_pb_trainer/dump_dataset:/workspace/dump_dataset \
           -v ~/output_trt_model:/workspace/output_trt_model \
           -v $(pwd):/workspace/source \
           -it jxnj502_experiment /bin/bash
```

## Usage

### Create json file

```shell
python inference.py --input_saved_model_path '~/.vaik-segmentation-trt-exporter/model.trt' \
                --input_classes_path './test_images/classes.txt' \
                --input_image_dir_path './test_images/raw' \
                --answer_image_dir_path './test_images/seg' \
                --output_dir_path '~/.vaik-segmentation-trt-experiment/test_images_out'
```

- input_image_dir_path
    - example

```shell
.
├── 000000005.png
├── 000000011.png
├── 000000017.png
├── 000000023.png
├── 000000029.png
├── 000000035.png
├── 000000041.png
├── 000000047.png
├── 000000053.png
├── 000000059.png
├── 000000065.png
├── 000000071.png
├── 000000077.png
├── 000000083.png
├── 000000089.png
└── 000000095.png

```

#### Output
- output_json_dir_path
    - example

```json
{
  "answer": {
    "array": [
      0,
      ・・・
      0
    ],
    "shape": [
      320,
      320
    ]
  },
  "labels": {
    "array": [
      0,
      ・・・
      0
    ],
    "shape": [
      320,
      320
    ]
  }
}
```
-----

### Calc mIoU

```shell
python calc_miou.py --input_json_dir_path '~/.vaik-segmentation-trt-experiment/test_images_out' \
                --input_classes_path './test_images/classes.txt' \
                --ignore_label 'background'
```

#### Output

``` text
mIoU:0.9512
background: 0.9987
zero: 0.9434
one: 0.9468
two: 0.9606
three: 0.9668
four: 0.9388
five: 0.9712
six: 0.9411
seven: 0.9377
eight: 0.9557
nine: 0.9502
```

-----

### Draw

```shell
python draw.py --input_json_dir_path '~/.vaik-segmentation-trt-experiment/test_images_out' \
                --input_classes_path './test_images/classes.txt' \
                --output_dir_path '~/.vaik-segmentation-trt-experiment/test_images_out_draw'
```

#### Output

![draw](https://user-images.githubusercontent.com/116471878/222900493-a2606e90-58c7-427e-9ae2-7d05b3e74e6b.png)
