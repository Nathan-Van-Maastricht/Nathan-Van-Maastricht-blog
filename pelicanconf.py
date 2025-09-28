AUTHOR = "Nathan Van Maastricht"
SITENAME = "Nathan Van Maastricht Blog"
SITEURL = ""

PATH = "content"

TIMEZONE = "Australia/Adelaide"

DEFAULT_LANG = "en"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
    ("LinkedIn", "https://www.linkedin.com/in/nathan-van-maastricht-72a177324/"),
    ("Github", "https://github.com/Nathan-Van-Maastricht"),
)

ARTICLE_URL = "articles/{date:%Y}/{date:%m}/{slug}.html"
ARTICLE_SAVE_AS = "articles/{date:%Y}/{date:%m}/{slug}.html"

PAGE_URL = "pages/{slug}.html"
PAGE_SAVE_AS = "pages/{slug}.html"

DEFAULT_PAGINATION = 10

THEME = "themes/Flex"

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

PLUGINS = ["render_math", "readtime", "sitemap"]
READ_TIME = True
READTIME_WPM = {
    "default": {
        "wpm": 200,
        "min_singular": "minute",
        "min_plural": "minutes",
        "sec_singular": "second",
        "sec_plural": "seconds",
    }
}

SITEMAP = {
    "format": "xml",
    "priorities": {"articles": 0.75, "indexes": 0.5, "pages": 0.95},
    "changefreqs": {"articles": "monthly", "indexes": "daily", "pages": "monthly"},
}
