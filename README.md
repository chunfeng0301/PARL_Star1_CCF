# PARL_Star1_CCF
DQN Paddle PARL
##  运行小trick
错误的设置evaluate.evl_step = 5000，得到最高平均分 dqn_model_7300(35.40).ckpt，然后改evl_step=1000，再训练，训练出模型：dqn_model_2100(7.92).ckpt；跑出804分
![img](https://github.com/chunfeng0301/PARL_Star1_CCF/blob/master/gifcc/star1.gif )  
![img](https://github.com/chunfeng0301/PARL_Star1_CCF/tree/master/gifcc/star1.gif)


## Requirements
All codes are tested under the following environment:
```shell
# or try: pip install -r requirements.txt

pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
pip install atari-py
pip install rlschool==0.3.1
```



##  跑分结果
```
100次共804分
[06-28 21:06:52 MainThread @machine_info.py:88] Cannot find available GPU devices, using CPU now.
[06-28 21:06:52 MainThread @machine_info.py:88] Cannot find available GPU devices, using CPU now.
[06-28 21:07:02 MainThread @Star1DQNEvl.py:309] step:0    eval_reward:7.980000000000001  
[06-28 21:07:13 MainThread @Star1DQNEvl.py:309] step:1    eval_reward:8.080000000000002  
[06-28 21:07:23 MainThread @Star1DQNEvl.py:309] step:2    eval_reward:8.02  
[06-28 21:07:34 MainThread @Star1DQNEvl.py:309] step:3    eval_reward:8.080000000000002  
[06-28 21:07:44 MainThread @Star1DQNEvl.py:309] step:4    eval_reward:8.02  
[06-28 21:07:55 MainThread @Star1DQNEvl.py:309] step:5    eval_reward:8.080000000000002  
[06-28 21:08:05 MainThread @Star1DQNEvl.py:309] step:6    eval_reward:8.02  
[06-28 21:08:15 MainThread @Star1DQNEvl.py:309] step:7    eval_reward:8.080000000000002  
[06-28 21:08:25 MainThread @Star1DQNEvl.py:309] step:8    eval_reward:8.02  
[06-28 21:08:36 MainThread @Star1DQNEvl.py:309] step:9    eval_reward:8.080000000000002  
[06-28 21:08:46 MainThread @Star1DQNEvl.py:309] step:10    eval_reward:8.02  
[06-28 21:08:56 MainThread @Star1DQNEvl.py:309] step:11    eval_reward:8.080000000000002  
[06-28 21:09:07 MainThread @Star1DQNEvl.py:309] step:12    eval_reward:8.02  
[06-28 21:09:17 MainThread @Star1DQNEvl.py:309] step:13    eval_reward:8.080000000000002  
[06-28 21:09:27 MainThread @Star1DQNEvl.py:309] step:14    eval_reward:8.02  
[06-28 21:09:38 MainThread @Star1DQNEvl.py:309] step:15    eval_reward:8.080000000000002  
[06-28 21:09:48 MainThread @Star1DQNEvl.py:309] step:16    eval_reward:8.02  
[06-28 21:09:58 MainThread @Star1DQNEvl.py:309] step:17    eval_reward:8.080000000000002  
[06-28 21:10:08 MainThread @Star1DQNEvl.py:309] step:18    eval_reward:8.02  
[06-28 21:10:19 MainThread @Star1DQNEvl.py:309] step:19    eval_reward:8.080000000000002  
[06-28 21:10:29 MainThread @Star1DQNEvl.py:309] step:20    eval_reward:8.02  
[06-28 21:10:39 MainThread @Star1DQNEvl.py:309] step:21    eval_reward:8.080000000000002  
[06-28 21:10:50 MainThread @Star1DQNEvl.py:309] step:22    eval_reward:8.02  
[06-28 21:11:00 MainThread @Star1DQNEvl.py:309] step:23    eval_reward:8.080000000000002  
[06-28 21:11:10 MainThread @Star1DQNEvl.py:309] step:24    eval_reward:8.02  
[06-28 21:11:20 MainThread @Star1DQNEvl.py:309] step:25    eval_reward:8.080000000000002  
[06-28 21:11:31 MainThread @Star1DQNEvl.py:309] step:26    eval_reward:8.02  
[06-28 21:11:41 MainThread @Star1DQNEvl.py:309] step:27    eval_reward:8.080000000000002  
[06-28 21:11:51 MainThread @Star1DQNEvl.py:309] step:28    eval_reward:8.02  
[06-28 21:12:02 MainThread @Star1DQNEvl.py:309] step:29    eval_reward:8.080000000000002  
[06-28 21:12:13 MainThread @Star1DQNEvl.py:309] step:30    eval_reward:8.02  
[06-28 21:12:23 MainThread @Star1DQNEvl.py:309] step:31    eval_reward:8.080000000000002  
[06-28 21:12:34 MainThread @Star1DQNEvl.py:309] step:32    eval_reward:8.02  
[06-28 21:12:44 MainThread @Star1DQNEvl.py:309] step:33    eval_reward:8.080000000000002  
[06-28 21:12:54 MainThread @Star1DQNEvl.py:309] step:34    eval_reward:8.02  
[06-28 21:13:05 MainThread @Star1DQNEvl.py:309] step:35    eval_reward:8.080000000000002  
[06-28 21:13:15 MainThread @Star1DQNEvl.py:309] step:36    eval_reward:8.02  
[06-28 21:13:26 MainThread @Star1DQNEvl.py:309] step:37    eval_reward:8.080000000000002  
[06-28 21:13:36 MainThread @Star1DQNEvl.py:309] step:38    eval_reward:8.02  
[06-28 21:13:47 MainThread @Star1DQNEvl.py:309] step:39    eval_reward:8.080000000000002  
[06-28 21:13:58 MainThread @Star1DQNEvl.py:309] step:40    eval_reward:8.02  
[06-28 21:14:08 MainThread @Star1DQNEvl.py:309] step:41    eval_reward:8.080000000000002  
[06-28 21:14:19 MainThread @Star1DQNEvl.py:309] step:42    eval_reward:8.02  
[06-28 21:14:30 MainThread @Star1DQNEvl.py:309] step:43    eval_reward:8.080000000000002  
[06-28 21:14:41 MainThread @Star1DQNEvl.py:309] step:44    eval_reward:8.02  
[06-28 21:14:51 MainThread @Star1DQNEvl.py:309] step:45    eval_reward:8.080000000000002  
[06-28 21:15:02 MainThread @Star1DQNEvl.py:309] step:46    eval_reward:8.02  
[06-28 21:15:13 MainThread @Star1DQNEvl.py:309] step:47    eval_reward:8.080000000000002  
[06-28 21:15:24 MainThread @Star1DQNEvl.py:309] step:48    eval_reward:8.02  
[06-28 21:15:36 MainThread @Star1DQNEvl.py:309] step:49    eval_reward:8.080000000000002  
[06-28 21:15:47 MainThread @Star1DQNEvl.py:309] step:50    eval_reward:8.02  
[06-28 21:15:58 MainThread @Star1DQNEvl.py:309] step:51    eval_reward:8.080000000000002  
[06-28 21:16:09 MainThread @Star1DQNEvl.py:309] step:52    eval_reward:8.02  
[06-28 21:16:20 MainThread @Star1DQNEvl.py:309] step:53    eval_reward:8.080000000000002  
[06-28 21:16:31 MainThread @Star1DQNEvl.py:309] step:54    eval_reward:8.02  
[06-28 21:16:42 MainThread @Star1DQNEvl.py:309] step:55    eval_reward:8.080000000000002  
[06-28 21:16:53 MainThread @Star1DQNEvl.py:309] step:56    eval_reward:8.02  
[06-28 21:17:04 MainThread @Star1DQNEvl.py:309] step:57    eval_reward:8.080000000000002  
[06-28 21:17:15 MainThread @Star1DQNEvl.py:309] step:58    eval_reward:8.02  
[06-28 21:17:26 MainThread @Star1DQNEvl.py:309] step:59    eval_reward:8.080000000000002  
[06-28 21:17:37 MainThread @Star1DQNEvl.py:309] step:60    eval_reward:8.02  
[06-28 21:17:47 MainThread @Star1DQNEvl.py:309] step:61    eval_reward:8.080000000000002  
[06-28 21:17:58 MainThread @Star1DQNEvl.py:309] step:62    eval_reward:8.02  
[06-28 21:18:08 MainThread @Star1DQNEvl.py:309] step:63    eval_reward:8.080000000000002  
[06-28 21:18:19 MainThread @Star1DQNEvl.py:309] step:64    eval_reward:8.02  
[06-28 21:18:29 MainThread @Star1DQNEvl.py:309] step:65    eval_reward:8.080000000000002  
[06-28 21:18:40 MainThread @Star1DQNEvl.py:309] step:66    eval_reward:8.02  
[06-28 21:18:50 MainThread @Star1DQNEvl.py:309] step:67    eval_reward:8.080000000000002  
[06-28 21:19:01 MainThread @Star1DQNEvl.py:309] step:68    eval_reward:8.02  
[06-28 21:19:11 MainThread @Star1DQNEvl.py:309] step:69    eval_reward:8.080000000000002  
[06-28 21:19:21 MainThread @Star1DQNEvl.py:309] step:70    eval_reward:8.02  
[06-28 21:19:32 MainThread @Star1DQNEvl.py:309] step:71    eval_reward:8.080000000000002  
[06-28 21:19:42 MainThread @Star1DQNEvl.py:309] step:72    eval_reward:8.02  
[06-28 21:19:53 MainThread @Star1DQNEvl.py:309] step:73    eval_reward:8.080000000000002  
[06-28 21:20:03 MainThread @Star1DQNEvl.py:309] step:74    eval_reward:8.02  
[06-28 21:20:13 MainThread @Star1DQNEvl.py:309] step:75    eval_reward:8.080000000000002  
[06-28 21:20:24 MainThread @Star1DQNEvl.py:309] step:76    eval_reward:8.02  
[06-28 21:20:34 MainThread @Star1DQNEvl.py:309] step:77    eval_reward:8.080000000000002  
[06-28 21:20:45 MainThread @Star1DQNEvl.py:309] step:78    eval_reward:8.02  
[06-28 21:20:55 MainThread @Star1DQNEvl.py:309] step:79    eval_reward:8.080000000000002  
[06-28 21:21:05 MainThread @Star1DQNEvl.py:309] step:80    eval_reward:8.02  
[06-28 21:21:16 MainThread @Star1DQNEvl.py:309] step:81    eval_reward:8.080000000000002  
[06-28 21:21:26 MainThread @Star1DQNEvl.py:309] step:82    eval_reward:8.02  
[06-28 21:21:37 MainThread @Star1DQNEvl.py:309] step:83    eval_reward:8.080000000000002  
[06-28 21:21:47 MainThread @Star1DQNEvl.py:309] step:84    eval_reward:8.02  
[06-28 21:21:58 MainThread @Star1DQNEvl.py:309] step:85    eval_reward:8.080000000000002  
[06-28 21:22:09 MainThread @Star1DQNEvl.py:309] step:86    eval_reward:8.02  
[06-28 21:22:19 MainThread @Star1DQNEvl.py:309] step:87    eval_reward:8.080000000000002  
[06-28 21:22:30 MainThread @Star1DQNEvl.py:309] step:88    eval_reward:8.02  
[06-28 21:22:41 MainThread @Star1DQNEvl.py:309] step:89    eval_reward:8.080000000000002  
[06-28 21:22:52 MainThread @Star1DQNEvl.py:309] step:90    eval_reward:8.02  
[06-28 21:23:02 MainThread @Star1DQNEvl.py:309] step:91    eval_reward:8.080000000000002  
[06-28 21:23:13 MainThread @Star1DQNEvl.py:309] step:92    eval_reward:8.02  
[06-28 21:23:24 MainThread @Star1DQNEvl.py:309] step:93    eval_reward:8.080000000000002  
[06-28 21:23:34 MainThread @Star1DQNEvl.py:309] step:94    eval_reward:8.02  
[06-28 21:23:45 MainThread @Star1DQNEvl.py:309] step:95    eval_reward:8.080000000000002  
[06-28 21:23:55 MainThread @Star1DQNEvl.py:309] step:96    eval_reward:8.02  
[06-28 21:24:06 MainThread @Star1DQNEvl.py:309] step:97    eval_reward:8.080000000000002  
[06-28 21:24:17 MainThread @Star1DQNEvl.py:309] step:98    eval_reward:8.02  
[06-28 21:24:28 MainThread @Star1DQNEvl.py:309] step:99    eval_reward:8.080000000000002  
[06-28 21:24:28 MainThread @Star1DQNEvl.py:314] test_reward_mean:8.049599999999998    test_reward_sum:804.9599999999998 
```
