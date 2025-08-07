async function getWeather() {
  const location = document.getElementById('locationInput').value;
  const resultBox = document.getElementById('weatherResult');

  if (!location) {
    resultBox.innerHTML = "❗ Please enter a location.";
    return;
  }

  const apiKey = "8a26deb6859541d0b3182935250504";
  const url = `http://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${location}&aqi=yes`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (data.error) {
      resultBox.innerHTML = `❌ Error: ${data.error.message}`;
    } else {
      const temp = data.current.temp_c;
      const condition = data.current.condition.text;
      const city = data.location.name;
      const country = data.location.country;

      resultBox.innerHTML = `
        📍 Location: <strong>${city}, ${country}</strong><br/>
        🌡 Temperature: <strong>${temp}°C</strong><br/>
        🌤 Condition: <strong>${condition}</strong>
      `;
    }
  } catch (error) {
    resultBox.innerHTML = "⚠️ Failed to fetch weather data. Try again later.";
    console.error(error);
  }
}
