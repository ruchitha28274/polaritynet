// Function to analyze sentiment
async function analyzeSentiment() {

    const reviewInput = document.getElementById("reviewInput").value;
    const resultDiv = document.getElementById("result");

    // If user didn't enter anything
    if (reviewInput.trim() === "") {
        resultDiv.innerHTML = "⚠️ Please enter a review.";
        return;
    }

    // Show loading message
    resultDiv.innerHTML = "⏳ Analyzing sentiment...";

    try {

        const response = await fetch("https://polaritynet-aykt.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: reviewInput
            })
        });

        const data = await response.json();

        let sentimentColor = "black";
        let icon = "😐";

        if (data.sentiment === "Positive") {
            sentimentColor = "green";
            icon = "😊";
        } 
        else if (data.sentiment === "Negative") {
            sentimentColor = "red";
            icon = "😡";
        } 
        else {
            sentimentColor = "black";
            icon = "😐";
        }

        resultDiv.innerHTML = `
            <h3 style="color:${sentimentColor}">
                ${icon} Sentiment: ${data.sentiment}
            </h3>
            <p>Confidence: ${data.confidence}%</p>
        `;

    } catch (error) {

        resultDiv.innerHTML = "❌ Error connecting to server.";

        console.error("API Error:", error);
    }
}
