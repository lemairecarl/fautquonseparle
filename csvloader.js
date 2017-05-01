function loadCsv(allText) {
    var allTextLines = allText.split(/\n/);
    var documents = {};

    lastId = -1;
    for (var i=0; i<allTextLines.length; i++) {
        var fields = allTextLines[i].split('\t');
        if (fields.length == 3) {
            fields[2] = $.trim(fields[2])
            documents[fields[0]] = {"id": fields[0], "q_id": fields[1], "text": fields[2]};
            lastId = fields[0];
        } else if (fields.length == 1) {
            // Inutile?
            documents[lastId]["text"] += "\n" + $.trim(fields[0]);
        } else {
            throw new Exception('Bad field count');
        }
    }
    return documents;
}