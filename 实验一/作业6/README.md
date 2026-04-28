 ORB 特征检测与目标定位实验
基于 OpenCV 实现 ORB 特征检测、匹配、RANSAC 优化、目标定位及参数对比实验

实验内容
1. ORB 关键点检测与可视化
2. ORB 特征匹配（BFMatcher）
3. RANSAC 剔除错误匹配
4. 基于单应矩阵的目标定位
5. nfeatures 参数对比实验（500/1000/2000）
6. ORB 与 SIFT 算法对比实验（选做）

运行环境
- Python 3
- OpenCV
- NumPy

运行方式
直接运行对应脚本即可自动完成：
- 图片自动识别与格式转换
- 特征检测、匹配、定位、绘图、结果保存

输出文件
- box_kp.png：模板特征点
- box_in_scene_kp.png：场景特征点
- orb_initial_matches.png：初始匹配
- orb_ransac_matches.png：RANSAC优化匹配
- target_detection_result.png：目标定位
- loc_result_nfeat_*.png：参数对比定位图
- sift_target_loc.png：SIFT定位结果

实验结论
- ORB 速度快，SIFT 鲁棒性更强
- 适中 nfeatures 可获得最佳匹配与定位效果
- RANSAC 可显著剔除误匹配
