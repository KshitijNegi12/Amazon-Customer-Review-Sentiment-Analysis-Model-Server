import scrapy
import numpy as np
from ..items import AmazonItem
from latest_user_agents import get_random_user_agent

class AmspySpider(scrapy.Spider):
	name = "amspy"
	allowed_domains = ["amazon.in"]
	start_urls = ["https://amazon.in"]

	custom_settings = {
		'FEEDS': {
			# write in file
			'./data.csv': {'format': 'csv', 'overwrite': True},
		}, 
		'DEFAULT_REQUEST_HEADERS': {
			# random fake user agent
            'User-Agent': get_random_user_agent(),
        },
	}

	def __init__(self,slink):
		self.slink=slink

	def start_requests(self):
		# search=input('enter: ')
		search = self.slink
		if len(search)>0:
			yield scrapy.Request(url=search, callback=self.parse_amazon)

# this works most of times because of same page reviews
	def parse_amazon(self,response):
		data = response.xpath('//*[@class="a-section celwidget"]/div/span/div/div[1]/span')
		for feed in data:
			li = feed.xpath('string()').get().strip()
			if li and not (li == '' or li == 'NaN' or li == np.nan):  
				item = AmazonItem()
				item['Review'] = li
				yield item	

# the next page which contain 10 reviews, not work now, since it wants cred to access
	# def parse_amazon(self, response):
	# 	# print(response.request.headers)
	# 	next_path = response.xpath('//div[@id="reviews-medley-footer"]/div[2]/a')
	# 	if next_path:
	# 		next_path = next_path.attrib['href']
	# 		next_url = response.urljoin(next_path)
	# 		yield response.follow(next_url, callback=self.follow_link)


	# def follow_link(self, response):
	# 	data=response.xpath('//div[@class="a-section a-spacing-none reviews-content a-size-base"]//div[@class="a-section review aok-relative"]//div[@class="a-row a-spacing-small review-data"]/span/span')
	# 	for feed in data:
	# 		li = feed.xpath('string()').get().strip()
	# 		item = AmazonItem()
	# 		item['text'] = li
	# 		yield item			
		
# This violate robot.txt, can't scrape further
		# next_page=response.xpath('//li[@class="a-last"]/a')
		# if next_page:
		# 	next_page_url=next_page.attrib['href']
		# 	next_page_url = response.urljoin(next_page_url)
		# 	yield response.follow(next_page_url, callback=self.follow_link)