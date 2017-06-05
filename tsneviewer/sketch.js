function preload() {
    jsondata = loadJSON('../fqsp.json');
}

function setup() {
    colorPalette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

    createCanvas(windowWidth, windowHeight);
    //words = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'];
    //vecs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]];
    words = jsondata.words;
    vecs = jsondata.vecs;
    q = jsondata.q;

    scale = 40.0;
    zoom = 1.0;
    zoomSpeed = 0.02;
    zoomExtent = [0.2, 20.0]

    windowOrigin = [windowWidth / 2, windowHeight / 2]
    worldOrigin = [0.0, 0.0]
}

function draw() {
    background('white');

    for(var i = 0; i < words.length; i++) {
        windowCoord = worldToWindow(vecs[i]);
        if (isInsideWindow(windowCoord)) {
            fill(colorPalette[q[i]]);
            text(
                words[i],
                windowCoord[0],
                windowCoord[1]
                );
        }
    }
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

    zoom += event.delta * zoomSpeed;
    zoom = constrain(zoom, zoomExtent[0], zoomExtent[1]);

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