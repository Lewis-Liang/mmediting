- [D] customed eval hooks：在每次测试后，保存PSNR、SSIM最优的模型，并记录最优的iter值
- [D] evaluation：每次验证的时候，使用所有的数据进行验证，而不是只使用输入的9帧
- [x] test：测试时，保存每一张帧的PSNR和SSIM，最后再输出平均值
- [D] 测试batch为1，accumulate gradient iter=8的情况
- [D] 清理没有加vggloss版本的实验结果
- [D] 测试fix_iter设置为2500
- [x] tools/test_multiple_times.sh
- [D] 测试hsvloss
- [x] histogram loss
- [x] ssim loss
- [D] 测试unet structure with vggloss and frame 7 or 9
- [x] 测试时候，batch的大小会不会影响最后性能？
- [D] 测试60k? 30k + 30k