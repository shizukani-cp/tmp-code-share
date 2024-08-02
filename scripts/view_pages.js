
const contents = [];

window.onload = () => {
    const pages_element = document.getElementById("pages");
    contents.forEach((content) => {
        const content_title_element = document.createElement("h3");
        const content_link_element = document.createElement("a");
        content_link_element.setAttribute("href", `contents/${content.href}`);
        content_link_element.appendChild(document.createTextNode(content.title));
        content_title_element.setAttribute("style", "font-size: 3rem")
        content_title_element.appendChild(content_link_element);
        pages_element.appendChild(content_title_element);
    });
};