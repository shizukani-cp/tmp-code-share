const LEVEL_AND_SIZE = {
    1:50,
    2:40,
    3:30,
    4:25,
    5:20,
    6:15
}
const SIDEBAR_ELEMENT = document.getElementById("sidebar");

for (let element of document.querySelectorAll("h1,h2,h3,h4,h5,h6")) {
    const level = element.tagName[1];
    element.setAttribute("name", element.innerText);
    const index_element = document.createElement("a");
    index_element.appendChild(document.createTextNode(element.innerText));
    index_element.setAttribute("style", `font-size: ${LEVEL_AND_SIZE[level]}px; color: #ffffff; text-decoration:none;`);
    index_element.setAttribute("href", `#${element.innerText}`)
    SIDEBAR_ELEMENT.appendChild(index_element);
}