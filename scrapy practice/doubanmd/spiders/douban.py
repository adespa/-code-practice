# -*- coding: utf-8 -*-
import scrapy
import json
import re
from ..items import DoubanItem, DoubancomItem


class DoubanSpider(scrapy.Spider):
    name = 'douban'
    allowed_domains = ['movie.douban.com']
    url_form = 'https://movie.douban.com/j/chart/top_list?type=11&interval_id=100%3A90&action=&start={}&limit=20'
    com_url = 'comments?start={}&limit=20&sort=new_score&status=P'
    page = 0
    com_page = 0
    start_urls = [url_form.format(page)]

    def parse(self, response):
        results = json.loads(response.text)
        Item = DoubanItem()
        for res in results:
            for field in Item.fields:
                Item[field] = res[field]
            yield Item
            yield scrapy.Request(Item['url']+self.com_url.format(self.com_page), callback=self.parse_comment)

        self.page += 20
        if self.page < 21:
            yield response.follow(self.url_form.format(self.page), self.parse)

    def parse_comment(self, response):
        ItemCom = DoubancomItem()
        for sls in response.css('div.comment'):
            ItemCom['usr'] = sls.css('span.comment-info a::text').extract_first()
            ItemCom['comment'] = sls.css('p::text').extract_first()
            yield ItemCom
        self.com_page += 20
        if self.com_page<101:
            yield response.follow(re.sub('start=(/d)&', response.url, self.com_page), self.parse_comment)


