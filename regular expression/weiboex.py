import re


text_path = '/Users/finogeeks/Documents/learn_datas/'
s_text = text_path + 'weibo.txt'
d_text = text_path + 'weibo_emo.txt'


def get_raw_data(sourse_file, dest_file):
    # pp = '@\w.*?:(.*?\[.*?)\\{2,}|\d{5,}'
    # pp1 = '</a>\s+?([^\@]*?\[[^\@]*?)\s+?(http | \d{5,})'
    f1 = open(dest_file, 'w', encoding='utf-8')
    idx = 0
    with open(sourse_file, 'r', encoding="utf-8") as f:
        while True:
            if idx % 10000 == 0:
                print('get lines:', idx)
            line = f.readline()
            if not line:
                break
            patter1 = '@\w.*?:([^:]*?\[[^:]*?)(//|\d{5,})'
            patter2 = '</a>\s*?([^:]*?\[[^:]*?)\s*?(\d{5,}|http|//)'
            results1 = re.findall(patter1, line, re.S)
            if results1:
                for res in results1:
                    pres = res[0].strip()
                    resl = len(re.sub('\[.*?\]', '', pres))
                    if resl > 2:
                        f1.write(pres)
                        f1.write('\n')
                        idx += 1
            results2 = re.findall(patter2, line, re.S)
            if results2:
                for res in results2:
                    pres = res[0].strip()
                    resl = len(re.sub('\[.*?\]', '', pres))
                    if resl > 2:
                        f1.write(pres)
                        f1.write('\n')
                        idx += 1
        f1.close()


def get_emo_dict(emo_file, dict_file):
    f1 = open(emo_file, 'r', encoding='utf-8')
    f2 = open(dict_file, 'w', encoding='utf-8')
    emo_dict = {}
    while True:
        line = f1.readline()
        if not line:
            break
        emos = re.findall('\[(.{1,3})\]', line)
        if not emos:
            print('faile')
            continue
        for emo in set(emos):
            if emo in emo_dict:
                emo_dict[emo] += 1
            else:
                emo_dict[emo] = 1
    emos = sorted(emo_dict.items(), key=lambda d: d[1], reverse=True)
    for (key, value) in emos:
        if value > 30:
            f2.write('{}   {}\n'.format(key, value))
    f1.close()
    f2.close()


# emo_file = text_path + 'weibo_emo.txt'
# dict_file = text_path + 'emo_dict.txt'
#
# get_emo_dict(emo_file, dict_file)
all_emo = []
emo_dict = emo_file = text_path + 'emo_dict_50.txt'
with open(emo_dict, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        all_emo.append(line.strip())

def get_final_data(raw_data, fin_data_x, fin_data_y):
    f1 = open(raw_data, 'r', encoding='utf-8')
    f2 = open(fin_data_x, 'w', encoding='utf-8')
    f3 = open(fin_data_y, 'w', encoding='utf-8')
    last_line = ''
    idx = 0

    while True:
        raw_line = f1.readline()
        if not raw_line:
            break
        emos = re.findall('\[(.{1,3})\]', raw_line)
        if not emos:
            continue
        nol_line = re.sub('\[.*?\]', '', raw_line)
        nol_line = re.sub('@.*?\s', '', nol_line.strip())
        r1 = u'[a-zA-Z0-9’,\._\@�#/（）:]+'
        noa_line = re.sub(r1, '', nol_line)
        ts_line = re.sub('[^\w\u4e00-\u9fff]+', '', noa_line.strip())
        if len(ts_line) < 3:
            continue
        # if len(noa_line) > 50:
        #     continue
        noa_line = re.sub(' {1,}', ' ', noa_line)
        if noa_line == last_line:
            print(noa_line)
            continue
        last_line = noa_line
        for emo in set(emos):
            if emo in all_emo:
                # print(nol_line)
                f2.write(noa_line.strip() + '\n')
                f3.write(emo + '\n')
                idx += 1

    f1.close()
    f2.close()
    f3.close()
    print(idx)


# raw_data = text_path + 'weibo_emo.txt'
# fin_data_x = text_path + 'weibo_emo_x.txt'
# fin_data_y = text_path + 'weibo_emo_y.txt'
#
# get_final_data(raw_data, fin_data_x, fin_data_y)

def get_emo_num(fin_data_y):
    pos = neg = nor = 0
    with open(fin_data_y, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            index = all_emo.index(line)
            if index < 11:
                pos += 1
            elif index>28:
                nor += 1
            else:
                neg += 1
    print('the positive bum:', pos, 'the negative num:', neg, 'the normal num:', nor)


fin_data_y = text_path + 'weibo_emo_y.txt'
get_emo_num(fin_data_y)
