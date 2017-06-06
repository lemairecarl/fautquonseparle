function preload() {
    jsondata = loadJSON('../fqsp.json');
}

function setup() {
    colorPalette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
    questionLabels = {
        3: "Indépendance: Comment se remettre en marche?",
        4: "Éducation: Comment permettre à tout le monde de réaliser son plein potentiel?",
        9: "Climat: Comment enclencher la transition?",
        0: "Démocratie: Comment reprendre le pouvoir?",
        1: "Économie: Comment développer le Québec selon nos priorités?",
        8: "Santé: Comment prendre soin de tout le monde?",
        5: "Premiers Peuples: Comment construire la solidarité entre nous?",
        7: "Culture: Comment favoriser une création artistique vivante et en assurer l’accès à tous?",
        6: "Diversité: Comment vivre ensemble sans racisme ni discrimination?",
        2: "Régions: Comment dynamiser toutes nos communautés?"
    };

    createCanvas(windowWidth, windowHeight);
    textAlign(CENTER, CENTER);
    rectMode(CENTER);
    noStroke();

    //words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];
    //vecs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]];
    textShort = jsondata.court;
    textLong = jsondata.long;
    vecs = jsondata.vecs;
    q = jsondata.q;
    ansId = jsondata['id'];

    // Shuffling the order of display for justice
    order = [];
    for(var i = 0; i < textShort.length; i++) {
        order[i] = i;
    }
    order = fyshuffle(order);

    scale = 40.0;
    zoom = 0.2;
    zoomSpeed = 0.01;
    zoomExtent = [0.2, 20.0]

    windowOrigin = [windowWidth / 2, windowHeight / 2]
    worldOrigin = [0.0, 0.0]

    shuffleWait = false;
    oldClosest = -1;
}

function draw() {
    background('white');

    var skip;
    if (zoom < 0.5) {
        skip = 8;
    } else if (zoom < 1.0) {
        skip = 4;
    } else if (zoom < 2.0) {
        skip = 2;
    } else {
        skip = 1;
    }

    for(var j = 0; j < order.length; j += skip) {
        var i = order[j];
        windowCoord = worldToWindow(vecs[i]);
        fill(210);
        rect(windowCoord[0], windowCoord[1], 20, 20);
    }

    var closestDist = 9999;
    var closest = 0;
    for(var j = 0; j < order.length; j += skip) {
        var i = order[j];
        windowCoord = worldToWindow(vecs[i]);
        if (isInsideWindow(windowCoord)) {
            distToMouse = sqDist(mouseCoord(), windowCoord);
            if (distToMouse < closestDist) {
                closestDist = distToMouse;
                closest = i;
            }
            drawWord(i);
        }
    }
    drawWord(closest, true);

    if (closest != oldClosest) {
        document.getElementById('qtext').innerHTML = questionLabels[q[closest]];
        document.getElementById('ans_id').innerHTML = ansId[closest];
        document.getElementById('question').setAttribute('style', 'border-left: 10px solid ' + colorPalette[q[closest]]);
        document.getElementById('reponse').innerHTML = textLong[closest];
    }
    oldClosest = closest;
}

function drawWord(i, hover=false) {
    windowCoord = worldToWindow(vecs[i]); //TODO optimiser
    if (hover) {
        fill('white');
        tw = textWidth(textShort[i]);
        rect(windowCoord[0], windowCoord[1], tw, textSize());
        fill('black');
    } else {
        fill(colorPalette[q[i]]);
    }
    text(
        textShort[i],
        windowCoord[0],
        windowCoord[1]
        );
}

function sqDist(a, b) {
    dx = a[0] - b[0];
    dy = a[1] - b[1];
    return dx * dx + dy * dy;
}

function isInsideWindow(coord) {
    return coord[0] > -200 && coord[0] < windowWidth + 200
        && coord[1] > -200 && coord[1] < windowHeight + 200
}

function worldToWindow(coord) {
    return [
        (coord[0] - worldOrigin[0]) * scale * zoom + windowOrigin[0],
        (coord[1] - worldOrigin[1]) * scale * zoom + windowOrigin[1]
    ];
}

function windowToWorld(coord) {
    return [
        (coord[0] - windowOrigin[0]) / (scale * zoom) + worldOrigin[0],
        (coord[1] - windowOrigin[1]) / (scale * zoom) + worldOrigin[1]
    ];
}

function mouseCoord() {
    return [mouseX, mouseY];
}

function mouseWheel(event) {
    // Shift the origins to the current mouse pos, so the zoom is natural
    worldOrigin = windowToWorld(mouseCoord());
    windowOrigin = mouseCoord();

    zoom -= event.delta * zoomSpeed;
    zoom = constrain(zoom, zoomExtent[0], zoomExtent[1]);

    if (event.delta > 0.0) {
        shuffleDisplay();
    }

    return false;
}

function windowResized() {
	resizeCanvas(windowWidth, windowHeight);
	windowOrigin = [windowWidth / 2, windowHeight / 2]
}

function mouseDragged() {
    windowOrigin[0] += mouseX - pmouseX;
    windowOrigin[1] += mouseY - pmouseY;
}

function shuffleDisplay() {
    if (shuffleWait) return;

    order = fyshuffle(order);
    shuffleWait = true;
    setTimeout(function(){ shuffleWait = false; }, 2000);
}

function fyshuffle(array) {
    // Fisher-Yates (from Mike Bostock)
    var counter = array.length;

    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        var index = Math.floor(Math.random() * counter);

        // Decrease counter by 1
        counter--;

        // And swap the last element with it
        var temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }

    return array;
}