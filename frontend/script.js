async function analyzeReview(){

let review = document.getElementById("reviewInput").value;

let response = await fetch("http://127.0.0.1:8000/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body: JSON.stringify({
text: review
})

});

let data = await response.json();

document.getElementById("sentimentResult").innerText =
"Sentiment: " + data.sentiment;

}