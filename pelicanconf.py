AUTHOR = 'Nathan Van Maastricht'
SITENAME = 'Nathan Van Maastricht Blog'
# SITEURL = "https://nathan-van-maastricht.github.io/Nathan-Van-Maastricht-blog"

PATH = "content"

TIMEZONE = 'Australia/Adelaide'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
    ("LinkedIn", "https://www.linkedin.com/in/nathan-van-maastricht-72a177324/"),
    ("Github", "https://github.com/Nathan-Van-Maastricht")
)

ARTICLE_URL = 'articles/{date:%Y}/{date:%m}/{slug}.html'
ARTICLE_SAVE_AS = 'articles/{date:%Y}/{date:%m}/{slug}.html'

PAGE_URL = "pages/{date:%Y}/{date:%m}/{slug}.html"
PAGE_SAVE_AS = "pages/{date:%Y}/{date:%m}/{slug}.html"

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True
