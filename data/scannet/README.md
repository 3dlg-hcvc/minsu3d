# ScanNet Instructions

To acquire the access to ScanNet dataset, Please refer to the [ScanNet project page](https://github.com/ScanNet/ScanNet) and follow the instructions there. You will get a `download-scannet.py` script after your request for the ScanNet dataset is approved:

```shell
python download-scannet.py -o data/scannet --type _vh_clean_2.ply
python download-scannet.py -o data/scannet --type _vh_clean.aggregation.json
python download-scannet.py -o data/scannet --type _vh_clean_2.0.010000.segs.json
```

Roughly 10.6GB free space is needed on your disk.
