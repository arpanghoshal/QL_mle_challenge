var points = null;
var xScale = null;
var yScale = null;
let lastButtonClicked = null;

let style = document.createElement('style');
style.innerHTML = `
  .disabled-container {
    pointer-events: none;
    opacity: 0.5;
  }

  .btn.selected {
    background-color: white;
    border: none;
    color: #007BFF;
    padding: 12px 30px;
    cursor: pointer;
    font-size: 20px;

  }
`;
document.head.appendChild(style);

window.onload = function () {

    // get the checkbox with id "toggle" and make it checked
    document.getElementById("toggle").checked = true;

    const pointSize = 5;

    var width = document.getElementById('centeredBox').offsetWidth;
    var height = document.getElementById('centeredBox').offsetHeight;

    // Create an SVG
    var svg = d3.select("#centeredBox").append("svg")
        .attr("width", width)
        .attr("height", height);

    // Create x scale
    xScale = d3.scaleLinear()
        .domain([-1, 1])
        .range([0, width]);

    // Create y scale
    yScale = d3.scaleLinear()
        .domain([1, -1])
        .range([0, height]);

    // Add a group for x axis
    svg.append("g")
        .attr("transform", "translate(0," + height / 2 + ")")
        .call(d3.axisBottom(xScale));

    // Add a group for y axis
    svg.append("g")
        .attr("transform", "translate(" + width / 2 + ",0)")
        .call(d3.axisLeft(yScale));

    // Add a group for the points
    points = svg.append("g")
        .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

    //svg.classed('disabled-container', true);

    var lastPoint = null;

    function follow_mouse(event) {
        // Check if the toggle switch is checked
        var toggleChecked = document.getElementById("toggle").checked;
        if (!toggleChecked) {
            return;
        }

        // Remove the last point if it exists
        if (lastPoint) {
            lastPoint.remove();
        }

        // Add a point at the mouse position
        var x = xScale.invert(d3.pointer(event, this)[0] - width / 2);
        var y = yScale.invert(0); // yScale.invert(d3.pointer(event, this)[1] - height / 2);

        // Add the new point directly to the points group
        lastPoint = points.append("circle")
            .attr("cx", xScale(x))
            .attr("cy", yScale(y))
            .attr("r", pointSize)
            .attr("fill", "black");
    
    }

    function add_point(event) {
        // Check if the toggle switch is checked
        var toggleChecked = document.getElementById("toggle").checked;
        if (!toggleChecked) {
            return;
        }

        // Add a point at the mouse position
        var x = xScale.invert(d3.pointer(event, this)[0] - width / 2);
        var y = yScale.invert(0);

        // Add the new point directly to the points group
        points.append("circle")
            .attr("cx", xScale(x))
            .attr("class", "point")
            .attr("cy", yScale(y))
            .attr("r", pointSize)
            .attr("fill", "black");
    }

    svg.on('mousemove', follow_mouse);
    svg.on('click', add_point);


    function drawLine(coordinates) {
        var lineGenerator = d3.line()
            .x(function (d) { return xScale(d.x); })
            .y(function (d) { return yScale(d.y); });

        svg.append("path")
            .datum(coordinates)
            .attr("class", "drawn-line")
            .attr("fill", "none")
            .attr("stroke", "#FF0000")
            .attr("stroke-width", 2)
            .attr("d", lineGenerator);
    }

    // Add event listener for "l" key press to draw a line
    window.addEventListener('keydown', function (event) {
        if (event.key === 'c') {
            
            // Clear all points
            points.selectAll(".point").remove();
        
            // Clear all lines
            svg.selectAll(".drawn-line").remove();

            var container = document.getElementsByClassName('blocked')[0];
            container.classList.remove('disabled-container');
            
            // Send clear request to server
            fetch("/clear", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                console.log(data);  // Logs the server response
            });
        }

        if (event.key === 'p') {


            // Clear all lines
            svg.selectAll(".drawn-line").remove();

            // console.log all points
            var e = getAllPointsCoordinates();

            // read the value of id="n_components"
            var n_components = document.getElementById("n_components").value;

            // read the value of id="epochs"
            var epochs = 1000; // document.getElementById("epochs").value;

            // create query string to /train endpoint with n_components and epochs
           // var query_string = "/train?n_components=" + n_components;
           // query_string += "&epochs=" + epochs;

            // create query string to /train or /update endpoint with n_components and epochs
            var query_string;
            if (lastButtonClicked === "normal") {
                query_string = "/train?n_components=" + n_components;
                var container = document.getElementsByClassName('blocked')[0];
                container.classList.add('disabled-container');

            } else if (lastButtonClicked === "online") {
                query_string = "/update?n_components=" + n_components;
                var container = document.getElementsByClassName('blocked')[0];
                container.classList.add('disabled-container');
            } else {
                return;
            }
            query_string += "&epochs=" + epochs;

            // access endpoint /train at localhost:8000, sending the data
            response = fetch(query_string, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ coordinates: e })
            }).then(function (response) {
                return response.json();
            }).then(function (data) {
                // extract pi, mu, sigma from data
                pi = data.pi;
                mu = data.mu;
                sigma = data.sigma;
                time = data.time;

                    // Display the time
                var timeDisplayElement = document.getElementById("time-display");
                 timeDisplayElement.innerText = "Time: " + time;

                // log data
                console.log("pi: " + pi);
                console.log("mu: " + mu);
                console.log("sigma: " + sigma);
                console.log("time: " + time);

                // compute the PDF for each x point
                // evenly space x points from -1 to 1
                var x = d3.range(-1, 1, 0.01);
                var pdf = x.map(function (d) { return computeGaussianMixturePDF(d, pi, mu, sigma); });

                // scale the pdf to the y axis
                var y_max = d3.max(pdf);
                var y_min = d3.min(pdf);
                var y_range = y_max - y_min;
                pdf = pdf.map(function (d) { return (d - y_min) / y_range; });

                // create the data for the line
                var coordinates = [];
                for (var i = 0; i < x.length; i++) {
                    coordinates.push({ x: x[i], y: pdf[i] });
                }

                // draw the line
                drawLine(coordinates);
            });




        }
    });

    function getAllPointsCoordinates() {
        // output list 
        var coordinates = [];

        // Select all the circles in the SVG
        const circles = d3.select("#centeredBox").selectAll(".point");

        // Loop through each circle and print its coordinates
        circles.each(function (d, i) {
            const xPixel = +d3.select(this).attr("cx"); // Convert to number
            const yPixel = +d3.select(this).attr("cy"); // Convert to number
            const xCoordinate = xScale.invert(xPixel + svg.attr("width") / 2);
            const yCoordinate = yScale.invert(svg.attr("height") / 2 - yPixel);

            console.log("Point " + (i + 1) + ": x = " + xCoordinate + ", y = " + yCoordinate);
            coordinates.push({ x: xCoordinate, y: yCoordinate });
        });

        return coordinates;
    }

}
document.getElementById("normalTrainBtn").addEventListener("click", function() {
    lastButtonClicked = "normal";
    this.classList.add("selected"); // add selected class
    document.getElementById("onlineTrainBtn").classList.remove("selected"); // remove selected class from the other button
});

document.getElementById("onlineTrainBtn").addEventListener("click", function() {
    lastButtonClicked = "online";
    this.classList.add("selected"); // add selected class
    document.getElementById("normalTrainBtn").classList.remove("selected"); // remove selected class from the other button
});

function computeGaussianMixturePDF(x, pi, mu, variance) {
    if (pi.length !== mu.length || pi.length !== variance.length) {
        throw new Error("pi, mu, and variance should have the same length.");
    }

    let pdf = 0;

    for (let i = 0; i < pi.length; i++) {
        const weight = pi[i];
        const mean = mu[i];
        const var_i = variance[i];

        const normalizingConstant = 1 / (Math.sqrt(2 * Math.PI * var_i));
        const exponent = -0.5 * Math.pow((x - mean), 2) / var_i;

        pdf += weight * normalizingConstant * Math.exp(exponent);
    }

    return pdf;
}


