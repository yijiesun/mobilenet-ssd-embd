echo userspace > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo 1416000> /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
echo 1416000> /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
echo 1416000> /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpu3/cpufreq/scaling_governor
echo 1416000> /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpu4/cpufreq/scaling_governor
echo 1800000> /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed
echo userspace > /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor
echo 1800000> /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed

cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq
cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq
