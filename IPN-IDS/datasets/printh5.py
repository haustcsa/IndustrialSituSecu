import h5py

# 打开H5文件
f = h5py.File('GasPipLine/gas_2-way30-shot_T6n180q_E2n60q_V8n35q/x_meta_train.h5', 'r')

for name in f:
    print(name)


print("Objects in the HDF5 file:")
for name in f:
    print(name)
    for dataset_name in f[name]:
        dataset = f[name][dataset_name]
        print(dataset_name, dataset.shape)  # 打印数据集名称和形状
        # 进一步操作和查看数据集内容


f.close()
