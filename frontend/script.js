async function analyze() {

let reviewText = document.getElementById("review").value;

if (reviewText.trim() === "") {
    alert("Please enter a review");
    return;
}

document.getElementById("loading").style.display = "block";
document.getElementById("result").innerHTML = "";

try {

    let response = await fetch("https://polaritynet-aykt.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: reviewText
        })
    });

    if (!response.ok) {
        throw new Error("API Error");
    }

    let data = await response.json();

    document.getElementById("loading").style.display = "none";

    let sentiment = data.final_sentiment;
    let confidence = data.confidence;
    let highlighted = data.highlighted_text;

    let icon = "";

    if (sentiment === "Positive") {
        icon = "😊";
    }
    else if (sentiment === "Negative") {
        icon = "😞";
    }
    else {
        icon = "😐";
    }

    document.getElementById("result").innerHTML =
    `<span class="${sentiment.toLowerCase()}">
        ${icon} ${sentiment} <br>
        Confidence: ${confidence}% <br><br>
        ${highlighted}
    </span>`;

}
catch (error) {

    document.getElementById("loading").style.display = "none";

    document.getElementById("result").innerHTML =
    `<span style="color:red;">
        Error connecting to API
    </span>`;

    console.log(error);
}

}
