from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor


class LaNacionSpiderPol(CrawlSpider):

    name = "la_nacion_politica"
    custom_settings = {
        'DOWNLOAD_DELAY': 4,  # Afectado por RANDOMIZE_DOWNLOAD_DELAY, (between 0.5 * DOWNLOAD_DELAY and 1.5 * DOWNLOAD_DELAY)
        'DEPTH_LIMIT': 3,
        'USER_AGENT': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:69.0) Gecko/20100101 Firefox/69.0',
        'COOKIES_ENABLED': False
    }

    # https://buscar.lanacion.com.ar/politica/c-Política/page-%d
    start_urls = ['https://buscar.lanacion.com.ar/politica/c-Política/page-%d' % n for n in range(1, 300)]

    rules = (
        Rule(LinkExtractor(allow=('https?://www.lanacion.com.ar/[0-9][0-9]+', )), callback='download'),
        Rule(LinkExtractor(allow=('https?://www.lanacion.com.ar/politica/.+nid[0-9][0-9]+', )), callback='download')
    )

    @staticmethod
    def download(response):
        page = response.url.split("/")[-1]
        filename = r'C:\Users\maxiu\Austral\OneDrive - AUSTRAL\Maestria\Segundo_Año\WM\wm_tp1\la_nacion\Politica\{}.html'.format(page)
        with open(filename, 'wb') as f:
            f.write(response.body)
        return
