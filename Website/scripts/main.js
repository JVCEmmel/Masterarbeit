let myImage = document.querySelector("img");

myImage.onclick = function() {
  let mySrc = myImage.getAttribute("src");
  if (mySrc == "./../LaTeX/Abbildungen/Abbildung_1_(acht1_028).jpg") {
    myImage.setAttribute("src", "./../LaTeX/Abbildungen/Abbildung_2_(acht1_066).jpg")
  } else {
    myImage.setAttribute("src", "./../LaTeX/Abbildungen/Abbildung_1_(acht1_028).jpg")
  }
}

let myButton = document.querySelector("button");
let myHeading = document.querySelector("h1");

function setUserName() {
  let myName = prompt("Please enter your name.");
  localStorage.setItem("name", myName);
  myHeading.textContent = "Mozilla is cool, " + myName;
}

if(!localStorage.getItem("name")) {
  setUserName();
} else {
  let storedName = localStorage.getItem("name");
  myHeading.textContent = "Mozilla is cool, " + storedName;
}

myButton.onclick = function() {
  setUserName();
}
