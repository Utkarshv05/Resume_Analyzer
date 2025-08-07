async function getNews() {
  const category = document.getElementById('categorySelect').value;
  const resultBox = document.getElementById('newsResult');

  if (!category) {
    resultBox.innerHTML = "❗ Please select a category.";
    return;
  }

  const apiKey = "46537d31990842e8b7086c2aeb956ccf";
  const url = `https://newsapi.org/v2/top-headlines?country=us&category=${category}&apiKey=${apiKey}`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (data.status !== "ok") {
      resultBox.innerHTML = "⚠️ Error fetching news.";
      return;
    }

    const articles = data.articles;

    if (articles.length === 0) {
      resultBox.innerHTML = "📭 No news found for this category.";
      return;
    }

    resultBox.innerHTML = "";
    articles.forEach(article => {
      const newsHTML = `
        <div class="news-item">
          <strong>${article.title}</strong><br/>
          <p>${article.description || "No description available."}</p>
          <a href="${article.url}" target="_blank">Read more 🔗</a>
        </div>
      `;
      resultBox.innerHTML += newsHTML;
    });
  } catch (error) {
    console.error(error);
    resultBox.innerHTML = "⚠️ Failed to fetch news. Try again later.";
  }
}
