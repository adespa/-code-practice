# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import json
from .items import DoubanItem, DoubancomItem

class DoubanmdPipeline(object):
    def open_spider(self, spider):
        self.file1 = open('moviefile.json', 'w', encoding='utf-8')
        self.file2 = open('moviecom.json', 'w', encoding='utf-8')

    def close_spider(self, spider):
        self.file1.close()
        self.file2.close()

    def process_item(self, item, spider):
        if isinstance(item, DoubanItem):
            line = json.dumps(dict(item), ensure_ascii=False) + '\n'
            self.file1.write(line)
            return item
        if isinstance(item, DoubancomItem):
            line = json.dumps(dict(item), ensure_ascii=False) + '\n'
            self.file2.write(line)
            return item
