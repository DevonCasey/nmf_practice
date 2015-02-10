## HTML Parsing

For this miniquiz we will get experience working with the web inspector in Chrome and working with HTML.  These will be important skills for today as we will be taking a deep dive into building interactive visualizations with D3.

___NOTE: HTML Parsing is distinct from Web scraping.  Often they are done at the same time, but I believe this conflates understanding and is an anti-pattern that does not respect abstraction.___

Assume we have previously downloaded the HTML of a bunch of Amazon products to use in a recommender.  Since parsing HTML in shell is quite difficult and painful (though not [impossible](http://www.w3.org/People/Raggett/tidy/)), we will use the powerful parsing capabilities of Python and its associated libraries.  Remember the defacto HTML parsing library for Python is [Beautiful Soup](http://www.crummy.com/software/BeautifulSoup/).

In the this folder, the HTML of a single Amazon product is included: `amazon.html`. Looking at the HTML file, you will see how much of a mess they are (Amazon has pretty complicated webpages ;).  It is much easier to use a web browser to find the correct section of the webpage to extract. 

1. Using a web browser with it's [developer tools](https://developers.google.com/chrome-developer-tools/) or [inspector](https://developer.mozilla.org/en-US/docs/Tools/Page_Inspector), find the class/id of the tag you need to extract to get the reviews.

2. Once you know what to extract, write a function using Beautiful soup to extract the text of the reviews for a single product (`amazon.html`). Your function should take a single string of HTML and return a list of strings (each string corresponding to a single review). Each block of text should have any leading/trailing newlines (\n) stripped from it.  _HINT: The reviews are nested in a `<div>` with id 'revMHRL'.  The text of each individual review is in a `<div>` with class 'reviewText'_

These are probably going to be helpful:

* [Seach by id](http://www.crummy.com/software/BeautifulSoup/bs4/doc/#the-keyword-arguments)
* [Search by class](http://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-by-css-class)
* [Extract text](http://www.crummy.com/software/BeautifulSoup/bs4/doc/#get-text)
