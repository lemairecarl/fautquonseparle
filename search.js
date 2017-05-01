/*
  Adapted from: http://jekyll.tips/jekyll-casts/jekyll-search-using-lunr-js/
*/

function displaySearchResults(results, csvdata) {
    var searchResults = document.getElementById('search-results');

    if (results.length) { // Are there any results?
      var appendString = '';

      for (var i = 0; i < results.length; i++) {  // Iterate over the results
        var text = csvdata[results[i].ref];
        appendString += '<li>' + text + '</li>';
      }

      searchResults.innerHTML = appendString;
    } else {
      searchResults.innerHTML = '<li>No results found</li>';
    }
}

function getQueryVariable(variable) {
    var query = window.location.search.substring(1);
    var vars = query.split('&');

    for (var i = 0; i < vars.length; i++) {
      var pair = vars[i].split('=');

      if (pair[0] === variable) {
        return decodeURIComponent(pair[1].replace(/\+/g, '%20'));
      }
    }
}

$(document).ready(function() {
    var searchTerm = getQueryVariable('query');

    if (searchTerm) {
        $.ajax({
            type: "GET",
            url: "fautquonseparle.txt",
            dataType: "text",
            success: function(data) {
                csvdata = loadCsv(data);

                idx = lunr(function () {
                    this.use(lunr.fr);
                    // then, the normal lunr index initialization
                    this.ref('id')
                    this.field('text');

                    for (var key in csvdata) {
                        this.add({"id": key, "text": csvdata[key]})
                    }
                });

                document.getElementById('search-box').setAttribute("value", searchTerm);

                var results = idx.search(searchTerm); // Get lunr to perform a search
                displaySearchResults(results, csvdata); // We'll write this in the next section
            }
        });
    }
});
