# coding=utf-8
import requests
import re
from requests.exceptions import RequestException
import pymongo


def open_one_page(kwords, pnum):
    url = 'https://zhidao.baidu.com/search?word=' + kwords + '&&pn=' + str(pnum)
    try:
        response = requests.get(url)
        response.encoding = "gb2312"
        if response.status_code == 200:
            return response.text
        else:
            return None
    except RequestException:
        print('request error......')


def parse_the_page(page_text):
    pattern = re.compile('<a href.*?"ti">(.*?)</a>.*?answer-text.*?</i>(.*?)</dd>', re.S)
    ques_ans = pattern.findall(page_text)
    for quan in ques_ans:
        yield {
            '问题': re.sub('</?em>', '', quan[0]),
            '回答': re.sub('</?em>', '', quan[1]),
        }


MONGO_URL = 'localhost'
MONGO_DB = 'baidu'
MONGO_TABLE = 'zhidao'
client = pymongo.MongoClient(MONGO_URL)
db = client[MONGO_DB]


def save_to_mongo(result):
    if db[MONGO_TABLE].insert(result):
        print('存储到MongoDB成功',result)
        return True
    return False


def main():
    page_content = open_one_page('投资', 60)
    # print(page_content)
    QAs = parse_the_page(page_content)
    for qas in QAs:
        save_to_mongo(qas)


if __name__ == '__main__':
    main()