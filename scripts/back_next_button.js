import { contents } from "./pages.js";

const main = document.querySelector("main");

const date_re = RegExp("(?<=contents/)(\d{8})/(?=entry.html)");
const date = parseInt(date_re.exec(location.pathname));

const entry_index = contents.findIndex((e) => { return e.date == date; });

if (entry_index !== 0) {
  const back_button = document.createElement("button");
  back_button.setAttribute("class", "button");
  back_button.setAttribute("id", "back");
  back_button.appendChild(document.createTextNode("前の記事"));
  back_button.onclick = () => {
    location.href = `../${contents[entry_index - 1].date}/entry.html`;
  };
  main.appendChild(back_button)
}

if (entry_index === (contents.length-1)) {
  const next_button = document.createElement("button");
  next_button.setAttribute("class", "button");
  next_button.setAttribute("id", "next");
  next_button.appendChild(document.createTextNode("次の記事"));
  next_button.onclick = () => {
    location.href = `../${contents[entry_index + 1].date}/entry.html`;
  };
  main.appendChild(next_button)
}
