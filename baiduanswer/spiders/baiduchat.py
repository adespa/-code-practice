# -*- coding: utf-8 -*-
# import scrapy
# from scrapy.spiders import CrawlSpider, Rule
# from scrapy.linkextractors import LinkExtractor
# from ..items import BaiduItem
#
#
# class BaiduchatSpider(CrawlSpider):
#     name = "baiduchat"
#     allowed_domains = ['zhidao.baidu.com']
#     start_urls = ['https://zhidao.baidu.com/search?word=%B9%C9%C6%B1+%D7%B7%CE%CA&ie=gbk&site=-1&sites=0&date=0&pn=0']
#
#     custom_settings = {
#         "ITEM_PIPELINES": {"baiduanswer.pipelines.BaiduPipeline": 300},
#     }
#
#     rules = (
#         Rule(LinkExtractor(allow=('date=0&pn=\d'), restrict_xpaths=('//a[@class="pager-next"]'))),
#         Rule(LinkExtractor(allow=('ie=gbk')), callback='parse_page')
#     )
#
#
#     def parse_page(self, response):
#         ans_num = len(response.css("div.line.content"))
#         print('0000000000000000')
#         Item = BaiduItem()
#         if (ans_num > 1):
#             title = response.css('span.ask-title::text').extract_first()
#             for ans in response.css("div.line.content"):
#                 res = self.extract_each_ans(ans)
#                 if res:
#                     res = title + '\n' + res
#                     Item['QAA'] = res[:-3]
#                     Item['LAB'] = res[-2:]
#                     yield Item
#
#     def extract_each_ans(self, ans):
#         if ans.css("div.ft-info grid"):
#             if ans.css("div.ft-info grid::text")[1].extract().strip() == '本回答由提问者推荐':
#                 rep_num = len(ans.css('div.replyask.line.replyask-ans'))
#                 if len(rep_num) > 1:
#                     content = []
#                     for rep in ans.css('div.replyask.line'):
#                         content += rep.css('div.ask.ask-supply::text').extract_first()
#                         content += ':'
#                         content += rep.css('div.replyask-content pre::text').extract_first().strip()
#                         content += '\n'
#                     content += ['正例\n']
#                     return content
#             else:
#                 rep_num = len(ans.css('div.replyask.line.replyask-ans'))
#                 eval_nums = ans.css('span.evaluate::attr(data-evaluate)').extract()
#                 if (len(rep_num) > 1) & (int(eval_nums[1]) - int(eval_nums[0]) > 1):
#                     content = []
#                     for rep in ans.css('div.replyask.line'):
#                         content += rep.css('div.ask.ask-supply::text').extract_first()
#                         content += ':'
#                         content += rep.css('div.replyask-content pre::text').extract_first().strip()
#                         content += '\n'
#                     content += ['负例\n']
#                     return content

from scrapy.spiders import Spider, Rule
import scrapy
from scrapy.linkextractors import LinkExtractor
from ..items import BaiduItem

with open('/Users/finogeeks/Documents/pyfile/baiduanswer/list.txt', 'r', encoding='utf-8') as f:
    word_list = []
    for line in f.readlines():
        word_list.append(line.strip())

class BaiduchatSpider(Spider):
    name = "baiduchat"
    allowed_domains = ['zhidao.baidu.com']
    # bas_url = 'https://zhidao.baidu.com/search?word={}%20追问'
    start_urls = ['https://zhidao.baidu.com/search?word={}%20股%20追问'.format(word) for word in word_list]
    pags = 0

    custom_settings = {
        "ITEM_PIPELINES": {"baiduanswer.pipelines.BaiduPipeline": 300},
    }


    def parse(self, response):
        for ques in response.css('div.list dl.dl'):
            q_url = ques.css('a::attr(href)').extract_first()
            yield scrapy.Request(q_url, callback=self.parse_page)
        next_page = response.css('a.pager-next::attr(href)').extract_first()
        if next_page:
            yield response.follow(next_page, callback=self.parse)
        # if next_page:
        #     self.pags += 1
        #     if self.pags < 20:
        #         yield response.follow(next_page, callback=self.parse)


    def parse_page(self, response):
        ans_num = len(response.css("div.line.content"))
        Item = BaiduItem()
        if (ans_num > 1):
            title = response.css('span.ask-title::text').extract_first()
            # print(title)
            for ans in response.css("div.line.content"):
                res = self.extract_each_ans(ans)
                if res:
                    # print(res)
                    res = '问:' + title + '\n' + res
                    Item['QAA'] = res[:-3]
                    Item['LAB'] = res[-3:]
                    yield Item

    def extract_each_ans(self, ans):
        if ans.css("div.ft-info.grid"):
            # print('no neg')
            # print('0000000000')
            if ans.css("div.ft-info.grid::text")[1].extract().strip() == '本回答由提问者推荐':
                rep_num = len(ans.css('div.replyask.line.replyask-ans'))
                eval_nums = ans.css('span.evaluate::attr(data-evaluate)').extract()
                print(eval_nums)
                if (rep_num > 1) & (int(eval_nums[0]) >= int(eval_nums[1])):
                    content = '答:' + ans.css('pre.best-text::text').extract_first() + '\n'
                    for rep in ans.css('div.replyask.line'):
                        # print(rep.css('div.ask.ask-supply::text').extract_first())
                        if rep.css('div.ask.ask-supply::text'):
                            # content += rep.css('div.ask.ask-supply::text').extract_first()
                            content += '问:'
                        if rep.css('div.reply.ask-supply::text'):
                            # content += rep.css('div.reply.ask-supply::text').extract_first()
                            content += '答:'
                        if rep.css('div.replyask-content pre::text'):
                            content += ' '.join(rep.css('div.replyask-content pre::text').extract()).strip()
                        content += '\n'
                    content += '正例\n'
                    return content
        else:
            rep_num = len(ans.css('div.replyask.line.replyask-ans'))
            eval_nums = ans.css('span.evaluate::attr(data-evaluate)').extract()
            # print(int(eval_nums[1]), int(eval_nums[0]))
            if (rep_num > 1) & (int(eval_nums[1]) >= int(eval_nums[0])):
                content = '答:' + ans.css('span.con::text').extract_first() + '\n'
                for rep in ans.css('div.replyask.line'):
                    # print(rep.css('div.ask.ask-supply::text').extract_first())
                    if rep.css('div.ask.ask-supply::text'):
                        # content += rep.css('div.ask.ask-supply::text').extract_first()
                        content += '问:'
                    if rep.css('div.reply.ask-supply::text'):
                        # content += rep.css('div.reply.ask-supply::text').extract_first()
                        content += '答:'
                    if rep.css('div.replyask-content pre::text'):
                        content += rep.css('div.replyask-content pre::text').extract_first().strip()
                    content += '\n'
                content += '负例\n'
                return content