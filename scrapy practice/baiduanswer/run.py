from scrapy import cmdline
name = 'baiduchat'
cmd = 'scrapy crawl {0}'.format(name)
cmdline.execute(cmd.split())