# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class DoubanItem(scrapy.Item):
    title = scrapy.Field()
    score = scrapy.Field()
    rank = scrapy.Field()
    url = scrapy.Field()

class DoubancomItem(scrapy.Item):
    usr = scrapy.Field()
    comment = scrapy.Field()