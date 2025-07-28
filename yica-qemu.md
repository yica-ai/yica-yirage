1. 获取系统镜像
首先获取qemu的系统镜像的基础版本，这个镜像已经安装好了rocm-5.7.3的版本，和其他一些rocm相关的驱动和库。
目录主要位置：
第一：10.12.70.52的机器是在/home/data/zhongjin.wu/image2/ubuntu22.04-kernel6.2.8_publish.img
各自拷贝上面的镜像到自己的目录下的image2中，并把镜像名字改为test2.qcow2，因为下面的脚本上使用的这个名字如下:
[Image]

2. 配置qemu
2.1 获取qemu二进制文件

第一种获取途径（后续以这个为主）
在服务器中自己根目录下输入下面命令，来获取软件包。
git clone -b g100-dev http://gitlab-repo.yizhu.local/release/software-release.git
#或者下面（二选一）
#注意使用下面，需要在gitlab上写入你自己的sshkey
git clone -b g100-dev git@gitlab-repo.yizhu.local:release/software-release.git
获取成功之后，目录如下：
[Image]
或者通过web进行访问下载，地址：http://gitlab-repo.yizhu.local/release/software-release/-/releases
效果如下：
[Image]



2.2 配置qemu的启动参数
在qemubin目录中有一个qemu2.sh文件，按照下面进行修改，如下：
[Image]

This content is only supported in a Feishu Docs
下载上面的qemu2.sh ，直接在文档可以下载
TAPNAME ：这个是tap的网卡名，按照自己的名字进行修改，如果你没有看到自己的tap，可能你没有加进来，联系你的直接领导，进行添加。
VNC_ADDR：VNC的配置ip地址和端口号。
MAC_ADDR：这个是mac地址，mac地址在同一个局域网是不能相同的，如果相同就冲突了。
MYBIN：这个按照你实际的地址填写，例如我的通过Git clone下载的，那么地址如下：
[Image]

上面两个地址按照<qemu-MAC和VNC端口地址分配 副本>的文档根据自己的姓名在表格中分配的信息填写，如果没有，联系你的直接领导进行添加。
例如：我吴仲金的名字在表格中的信息如下：
[Image]
tapname是wzjtap0~3就是说可以使用wzjtap0，wzjtap1，wzjtap2，wzjtap3任何一个，
vnc端口号16~19，就是VNC_ADDR:这个参数10.12.70.52:16，其中ip地址是host的ip地址，最后的16是端口号，16~19都是你的。
MAC地址也是一样，每个人有4个地址可以使用。
如果上面表格没有你对应的名字，说明你新来的员工，还未加入，找你的直接领导去申请一下。
2.3 启动脚本
chmod +x qemu2.sh
./qemu2.sh
起来后，敲入info version 出现如下：
[Image]

[Image]

面显示的版本信息，应该和你下载的qemubin版本一致才对。
说明系统已经成功启动了，可以连接VNC查看系统的ip地址
如果出现tap网口没有，找你直接领导，将你的名字加入服务器中。
2.3.1.通过启动脚本配置cimdie 和cluster数量
qemu默认状态下支持8个cimdie，每个cimdie拥有4个cluster(最大值)。
如果想修改cimdie和cluster数量，可以通过在启动脚本中添加cimdie_cnt和cluster_cnt参数来实现。
[Image]
2.4 gem5和qemu一起启动方法
gem5是一个模拟器，risc-V的模拟器，如果想要跑算子加载到CIM die中，就需要启动gem5。
具体的环境搭建参见<SIMU 环境搭建 副本>。
2.4.1 下载启动包：
下载 使用Git clone  git@10.12.70.33:release/software-release.git，使用最新的版本见如下：
[Image]
启动脚本使用如下：
This content is only supported in a Feishu Docs
This content is only supported in a Feishu Docs
下载上面2个文件，直接在文档可以下载。放在同一个目录下，修改参数修改成自己的路径，具体见2.2小节。

2.4.2 gem5自动运行方法：
把这2个文件加上+x，再运行qemu2.sh 就可以了，如下：
chmod +x gem5.sh
chmod +x qemu2.sh
./qemu2.sh
注意：上面的gem5的脚本依赖gem5的运行环境库(环境安装见SIMU 环境搭建 副本)，二进制文件下载是如下地址：
从服务器10.11.60.100的/opt/tools/gem5-release，下载下来gem5的二进制包，放到自己所在的服务器相同目录，既/opt/tools/下
[Image]
上面gem5中的脚本，要根据实际路径填写
[Image]

2.4.3 gem5需要手动运行方(建议使用这个方式启动)：
在qemu2.sh中注释掉./gem5.sh ${UNIX_FILE} & 和sleep行，如下：
[Image]

然后需要开启2个终端一个运行gem5，一个运行qemu2.sh,如下：
#第一个终端执行
./gem5.sh /tmp/zhongjin.wu #这zhongjin.wu使用你自己的名字，并且和qemu2.sh里面的${UNIX_FILE}变量一致

#第二终端执行
./qemu2.sh

2.4.4 不启动gem5方法（在手动运行的基础上）
如果你不需要跑算子加载到cim die中运行，可以不启动gem5的脚本，但qemu2.sh的脚本需要修改一下（在手动启动基础上进行修改），否则会启动不了qemu2.sh，报错误。
只需要修改qemu2.sh中的rp=on，改成rp=off，如下：
[Image]
2.5 双卡和多卡情况下gem5和qemu的启动方式：
双卡、多卡的启动方式和2.4小节差异不大，只是一小部分修改就可以了，如下：
下面以三卡举例子，
下载程序包，见2.4.1小节，这里就不叙述了。
启动脚本如下：
This content is only supported in a Feishu Docs
这个启动脚本还是按照2.2进行修改成自己的目录和镜像路径等。
然后再运行，就可以得如下：
[Image]
虚拟机起来之后，ssh登入，输出rocm-smi 和 lspci -tv查看如下：
[Image]
上面已经显示三卡正常了。
上面启动脚本主要注意如下：
[Image]
socket文件，按照你需要几张卡，就创建几个，这里是3卡，那就创建3个，并按序号修改。
[Image]
gem5的启动，按照上面多少个文件，就启动几个，上面是3个这里就三个，超过三个就自己添加，并修改文件的序号（下图红框位置）。
[Image]
这里是qemu的启动参数，上面红框的位置，需要按序号添加，当然这里几个要和gem5启动次数一致，最多可以添加8个。
下面举例一下增加第4个的情况，如下：
[Image]

上面是gem5自动启动的方式，手动启动方式见2.4.3一样，只不过这里3卡就需要3个终端分别启动gem5，并指定的目录和qemu脚本${UNIX_FILE1}参数一致。

3. 配置VNC
3.1 下载vnc viewer并安装如下：

This content is only supported in a Feishu Docs
3.2 端口号填写
VNC_ADDR="10.12.70.52:16"：这里的ip地址按照你host系统的ip地址填写，后面的端口16是连接vnc的使用的（这里的端口号每个人分配的不一样，具体见qemu-MAC和VNC端口地址分配 副本），对应的是VNC的端口是5916（5900+16，根据表格的端口号加上5900），在vnc连接的时候，填5916的端口号，具体如下图：
[Image]

启动之后如下：
3.3 查看ip地址-命令行
进入系统如果是命令行模式的，就敲入
ifconfig
[Image]
3.4 查看IP地址-图形方式
查看上面的ip地址，如果大家对命令行不熟悉，可以启动图形界面，上面的镜像默认是命令行模式，需要启动图形敲入下面的命令
sudo systemctl start gdm.service 
[Image]
网络的部分一定要改为自动，如果手动会出现ip冲突的问题。
[Image]
从手动修改成自动需要断开网络，再连接一下，新的ip地址才获取到。
[Image]
4. 配置SSH
4.1 下载Xshell并安装
This content is only supported in a Feishu Docs
上面的压缩包解压之后如下：
[Image]
XmanagerPowerSuite-7.0.0004r.exe安装文件
上面安装完成会生成
[Image]
进入里面可以看到
[Image]
红框就是Xshell,双击就可以启动了。
Patch是破解文件夹，上面安装完成把这里面的文件拷贝到xshell的安装目录（默认的目录C:\Program Files (x86)\NetSarang\Xshell 7），再执行破解
[Image]
查看破解是否成功，打开Xshell软件
点击 帮助->关于Xshell， 可以看到下面表示破解成功
[Image]
4.2 连接SSH
打开xshell界面
[Image]
新建会话
填入下面的信息
[Image]
IP地址根据前面虚拟机获取到IP地址填写
点击用户身份验证
[Image]
填入用户名和密码，镜像默认的
用户名：yizhu
密码：yizhu
这样就可以连接了。
5. 配置git的选项
此镜像已经安装有git软件，只需要进行一些配置就可以使用Git clone的命令
使用ssh登入之后进行下面的操作。
5.1 修改gitconfig配置文件
在yizhu的根目录，打开.gitconfig文件，文件中的出现xxxx改为自己的信息，如下结果；
vi .gitconfig
[Image]
把xxxx改成自己的名字和邮箱地址
5.2 修改commit的提交模版
vi .git-commit-template.txt
[Image]
xxxx改成上面红框一样，改成自己的名字。
5.3 添加gitlab的秘钥
生成sshkey秘钥

mkdir -p .ssh
ssh-keygen
上面遇到需要填写的信息，可以不填直接回车就好
cat .ssh/id_rsa.pub
复制上面的id_rsa.pub文件信息到gitlab的ssh秘钥
[Image]
最后点击添加秘钥
这样gitlab的秘钥添加好，后续可以直接使用git clone git@10.12.70.52:/xxxx/xxx/xxx/就无需每次输入密码了，
但如果使用http的格式还是需要输入密码的。

5.4 使用glog命令查看版本信息
敲入命令glog
[Image]
在qemu中敲入 info version ，对应版本号如下：
[Image]
上面的版本号和git代码版本就可以对应上了，方便代码有问题的时候进行有效的沟通。



6. qemu、驱动、runtime库升级方法
6.1 拷贝升级包
qemu程序的升级见2.1节。
一、通过下面进行下载
git clone -b g100-dev http://gitlab-repo.yizhu.local/release/software-release.git
#或者下面（二选一）
#注意使用下面，需要在gitlab上写入你自己的sshkey
git clone -b g100-dev git@gitlab-repo.yizhu.local:release/software-release.git
二、通过web进行访问下载，
地址：http://gitlab-repo.yizhu.local/release/software-release/-/releases
[Image]
优先使用第二种，这样空间会小很多，第一个全部下载会很大30多G。

6.2 执行升级脚本
解压之后里面有一个install.sh命令，使用sudo执行就好了，执行成功之后。系统会自动加载驱动
sudo ./upgrade.sh 

[Image]
重启成功之后查看一下驱动是否已经是新的了
使用如下命令查看，并和packet_bin/ko目录的文件进行对比，是否一样
安装包里面的ko的版本信息：
[Image]

代码打印中的版本信息：
#第一种查看驱动的版本信息（以这个为准）
cat /sys/class/drm/card1/device/code_version
#第二种查看驱动的版本信息
sudo dmesg | grep yz-g100 


[Image]

可以看到上面2个是一致的，说明升级成功了，至此升级完成。

7. qemu内部的磁盘快照
7.1 创建快照
运行以下命令：
qemu-img snapshot -c snapshot_name vm_name.qcow2
其中，snapshot_name是您想要给快照命名的名称，vm_name.qcow2是要创建快照的虚拟机磁盘镜像的名称。该命令将创建一个名为snapshot_name的新快照。
使用快照就按之前能够正常运行的处理，你可以尝试添加和删除几个文件。
7.2 恢复快照
如果您想恢复到之前创建的快照，运行以下命令：
qemu-img snapshot -a snapshot_name vm_name.qcow2
其中，snapshot_name是您之前创建的快照的名称，vm_name.qcow2是虚拟机磁盘镜像的名称。该命令将使虚拟机恢复到创建快照时的状态。
7.3 删除快照
qemu-img snapshot -d snapshot-name vm_name.qcow2
qemu-img.exe snapshot -d snapshot-name vm_name.qcow2
7.4 查看镜像中保存的快照列表
qemu-img snapshot -l vm_name.qcow2
注意：尽量不要在虚拟机在运行的状态下来操作磁盘快照啊，可能会损坏的。

8. 配置Zsh环境
上面镜像的默认的时候bash的SHELL环境，但保留了zsh的环境，只是没有启用。
如果喜欢zsh环境的小伙伴，自己可以配置一下。
只要打开.bashrc，把最后两行的#号去掉，就能生效zsh了。
[Image]
修改完成并保存，然后退出ssh，重新登入就可以使用zsh了，如下：
[Image]

9. qemu的gdb调试
在qemu的启动参数中加入gdb --args，如下：
[Image]
然后运行qemu的启动脚本，启动之后，运行run命令
[Image]
当然也可以加断点什么的，和正常gdb操作是一样的，这里就不细讲，如果不懂gdb调试，自己找相关资料查询。
9.1 出现SIGUSR1的信号
目前52和53服务器，运行gdb会出现SIGUSR1的信号的干扰。
需要run之前加入下面一句话：
(gdb) handle SIGUSR1 nostop noprint
再运行就好了，如下：
[Image]

或者使用在gdb 运行的时候加入参数，这样就不需要每次进来都输入上面一遍，如下：
gdb --eval-command="handle SIGUSR1 nostop noprint"  --args  $MYBIN/qemu-system-x86_64 \
[Image]


10. 修改qemu虚拟机镜像文件大小
以下命令需要切换到root进行操作
10.1 查看kvm镜像的格式信息
注意：下面的命令是在本地主机运行下面的命令
[root@localhost kvm_a]# qemu-img info kvm_a.qcow2
image: kvm_a.qcow2
file format: qcow2
virtual size: 200G (214748364800 bytes)
disk size: 88G
cluster_size: 65536
kvm_a.qcow2 这个是qemu镜像的名字及其路径，需要替换自己的镜像路径名字

10.2 给镜像硬盘增加50G空间
注意：下面的命令是在本地主机运行下面的命令
[root@localhost kvm_a]# qemu-img resize kvm_a.qcow2 +50G
Image resized.
[root@localhost kvm_a]# qemu-img info kvm_a.qcow2
image: kvm_a.qcow2
file format: qcow2
virtual size: 300G (322122547200 bytes)
disk size: 88G
cluster_size: 65536

kvm_a.qcow2 这个是qemu镜像的名字及其路径，需要替换自己的镜像路径名字

10.3 扩大新硬盘的分区表--命令行
注意：下面的命令是进入qemu镜像系统里面运行下面的命令
1> parted [/dev/sdX]
#选择要修改分区的硬盘名 例如：parted /dev/sda
unit s 
#设置以扇区（sector）为单位
print free
#查看空闲空间
resizepart [partition number] [end]
#扩展分区大小
quit
2> partprobe /dev/sda
#刷新分区
3> resize2fs /dev/sda3
#重置 /dev/sda3 的空间大小

需要再root权限下使用
下面举例说明，具体的实际步骤
10.3.1 第一步：fdisk
fdisk -l
[Image]
上面./dev/sda 容量150G了，但是sda3还是100G，那是因为我们的硬盘扩大到了150G，但分区还没有扩大。

10.3.2 第二步：Parted
# parted [/dev/sdX]
parted /dev/sda
[Image]
提示是否修复，选择Fix（修复）
10.3.3 第三步：free
#设置以扇区（sector）为单位
unit s
#查看空闲空间
print free
切换显示单元变成扇区数，并打印出free空间，如下图：
[Image]
10.3.4 第四步：resizepart
#resizepart [partition number] [end]
resizepart 3 314572766
print free
[Image]
可以看到上面的free空间已经没有了，合并到3分区中了。（注意314572766，这个数字按照自己的大小来改）
#退出 parted 命令
quit
10.3.5 第五步：resize2fs
#刷新分区
partprobe /dev/sda
#重置 /dev/sda3 的空间大小
resize2fs /dev/sda3
[Image]
#检查分区是否已经生效
df -h
[Image]
上面的/dev/sda3由原来的100G增加到147G，这里无法增加到50G，因为ext4格式需要消耗一些空间的。


11. 常见问题：
11.1 问题1：qcow2镜像中创建新内容会变大，但是删除却不会自动缩小。
原因：
linux所谓的删除文件，只是将文件做了标记，并不是真正的删除。
处理方案：
11.1.1 第一步：删除用不到的文件
先在虚拟机中删除用不到的文件.
11.1.2 第二步：删除标记的文件
将做了删除标记的文件全部覆盖，然后重新使用rm -f 删除！
 dd if=/dev/zero of=/zero.dat 
#时间长的很 
rm /zero.dat -f

关闭虚拟机，退出qemu镜像
11.1.3 第三步：执行镜像转存命令
qemu-img convert -O qcow2 debian.qcow2 debian_new.qcow2
#时间长的很
或者
可以添加 -c 进行压缩:
qemu-img convert -c -O qcow2 debian.qcow2 debian_new.qcow2
#时间比较长
该压缩会将某些文件改为只读的，当有写入时会自动解压。

11.2 问题2：如何改变内核的版本
我们的qemu和驱动要依赖内核版本为6.2.8的版本，有一些自己自己制作的镜像版本，由于没有取消Ubuntu的系统自动更新功能，时间长了，忽然发现驱动无法使用了，一查看系统的内核版本被改了
sudo update-grub
[Image]
发现多了几个内核版本
现在需要做的是
11.2.1 取消系统的自动更新功能
ubuntu中关闭自动更新的方法：
1、打开终端；
2、输入命令打开/etc/apt/apt.conf.d/10periodic配置文件；
3、在配置文件中修改设置进行关闭即可。
具体操作步骤：
1、使用快捷键打开终端；
2、输入以下命令打开/etc/apt/apt.conf.d/10periodic配置文件。
vi etc/apt/apt.conf.d/10periodic
3、在配置文件中修改以下设置进行关闭即可。
#0是关闭，1是开启，将所有值改为0
APT::Periodic::Update-Package-Lists "0";
APT::Periodic::Download-Upgradeable-Packages "0";
APT::Periodic::AutocleanInterval "0";
相关操作：
在图形界面中，可通过以下操作步骤关闭。
1、菜单栏点击“系统”→ 选择“首选项”→ 启动应用程序 → 更新提示。
2、然后在弹出的窗口中将选项前面的钩去掉，实现从不更新，保存设置关闭即可。
11.2.2 更改启动grub指定到6.2.8的内核版本。
sudo update-grub
[Image]
从上图可以看到6.2.8内核之上增加了4个，所以6.2.8内核对应的第4个。
sudo vi /etc/default/grub
修改GRUB_DEFAULT这个一行，改成GRUB_DEFAULT="1> 4"这个里的4要根据上面的update-grub命令得到的数据6.2.8之前的增加了几个就填几，像上面是4个，那这里就填4。
[Image]
修改完成之后，再次执行
sudo update-grub
执行成功之后，重启系统之后，就可以回到6.2.8内核。

11.3 问题3：qemu异常死机，无法正常退出
另外如果vnc登录异常，QEMU的进程会卡死，需要手动杀死进程。
先用命令查看进程
ps -aux | grep qemu
找到自己用户名的进程
kill -9 进程id

11.4 问题4：疯狂打印IPV6的检测信息
Dmesg 信息中疯狂打印IPV6的检测信息，如下：
[Image]
解决办法：
通过vnc进入之后，在终端中，输入用户名和密码之后，再输入下面命令，打开图形界面：
sudo systemctl start gdm.service
进入图形界面之后，点击网络按钮->设置，
关闭ipv6，如下图。
[Image]
然后网络断开，再连接，让设置生效一下。

11.5 问题5：清除镜像空间无效占用
11.5.1 进入var目录
使用root权限执行如下：
cd /var
du -h -d 1 ./
[Image]
发现lib和log目录占用空间大
11.5.2 进入lib目录
执行下面命令进一步目录的大小
 cd /var/lib
 du -h -d 1 ./| grep G
[Image]
发现docker目录为最大空间消耗：
docker images
[Image]
删除docker的所有镜像，命令如下：
docker rmi -f yz_rocm:latest
docker rmi -f rocm/rocm-terminal:5.7
#如果还有其他镜像，继续删除
#注意这个命令一定要在全部镜像删除完之后再执行
rm -rf /var/lib/docker/overlay2/*
11.5.3 进入上面/var/log目录
查看具体的那个文件的大小
cd /var/log
ls -alh |grep G
[Image]
发现上面2个文件比较大，这2个文件清空内容，但不要删除，命令如下：
rm -rf *.gz
echo > syslog
echo > syslog.1
echo > kern.log
echo > kern.log.1
上述2步就可以释放很多空间。
如果还觉得不够，可以继续查看是其他目录，进行同样的方法进行删除操作。
上面删除之后，时间长了还是会继续增加的，现在搞一个一劳永逸的方法，限制这个两个文件的大小。
11.5.4 限制syslog、kern.log 文件的大小
打开/etc/logrotate.d/rsyslog文件
sudo vi /etc/logrotate.d/rsyslog
在文件最下面加入下面代码
{
    size 100M
    rotate 4
    compress
    postrotate
        /usr/lib/rsyslog/rsyslog-rotate
    endscript
}
效果如下：
[Image]

然后执行下面命令，就能生效或者重启系统也可以。
sudo logrotate /etc/logrotate.conf --force
11.5.5 进入/var/log/journal
[Image]

上面图片的journal目录最大，直接删除journal下面的文件，但重启之后， 又会回来，需要执行下面命令：
#journalctl 命令自动维护文件大小
#只保留近一周的日志
journalctl --vacuum-time=1w
#只保留 500MB 的日志
journalctl --vacuum-size=500M
这个这个目录最大就是有500M了，执行如下：
[Image]
11.5.6 限制journal目录的空间大小
sudo vi /etc/systemd/journald.conf
按照如下设置：
[Image]
再执行下面命令
sudo systemctl restart systemd-journald.service


11.6 问题6：无法进入系统镜像，提示initramfs
现象如下：
[Image]
这个情况就是镜像系统，出现文件格式校验错误，导致系统无法进入，解决的办法就是修复一下。
修复命令：
fsck -y /dev/sda3
直到修复完成，重启一下，就可以正常进入了。
