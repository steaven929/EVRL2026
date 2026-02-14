import os


def convert_to_sumo_network(typ_file, raw_net_file, output_net_file):
    # SUMO的netconvert命令，生成SUMO路网
    command = f"netconvert --type-files={typ_file} --output-file={output_net_file} --net-file={raw_net_file}"

    # 执行命令
    os.system(command)


# 输入文件路径
typ_file = "E:/AAA研1/9.14电子狗/bigemap使用说明/test.typ.xml"
raw_net_file = "E:/AAA研1/9.14电子狗/bigemap使用说明/test_raw.net.xml"
output_net_file = "E:/AAA研1/9.14电子狗/bigemap使用说明/output.net.xml"

# 执行转换
convert_to_sumo_network(typ_file, raw_net_file, output_net_file)
