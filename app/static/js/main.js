// Function to show error message
function showError(message) {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    } else {
        alert(message);
    }
}

// Function to hide error message
function hideError() {
    const errorDiv = document.getElementById('error-message');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// Function to update results
function updateResults(data) {
    const resultsDiv = document.getElementById('prediction-results');
    const chartImg = document.getElementById('prediction-chart');
    const resultsTable = document.getElementById('results-table');
    
    if (data.success) {
        // Update chart
        chartImg.src = 'data:image/png;base64,' + data.plot_url;
        chartImg.style.display = 'block';
        
        // Update table
        if (resultsTable) {
            const tbody = resultsTable.querySelector('tbody');
            tbody.innerHTML = '';
            
            const years = Object.keys(data.predictions.lstm);
            years.forEach(year => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${year}</td>
                    <td>${data.predictions.lstm[year].toFixed(2)}%</td>
                    <td>${data.predictions.xgboost[year].toFixed(2)}%</td>
                `;
                tbody.appendChild(row);
            });
            
            resultsTable.style.display = 'table';
        }
        
        resultsDiv.style.display = 'block';
        hideError();
    } else {
        showError(data.error || 'An error occurred while generating predictions');
        resultsDiv.style.display = 'none';
    }
}

// Handle form submission
document.getElementById('prediction-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const country = document.getElementById('country').value;
    const sex = document.getElementById('sex').value;
    const ageGroup = document.getElementById('age-group').value;
    
    // Validate inputs
    if (!country || !sex || !ageGroup) {
        showError('Please select all required fields');
        return;
    }
    
    // Show loading state
    const submitButton = this.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.textContent = 'Loading...';
    
    // Hide previous results and errors
    document.getElementById('prediction-results').style.display = 'none';
    hideError();
    
    // Send prediction request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            country: country,
            sex: sex,
            age_group: ageGroup
        })
    })
    .then(response => response.json())
    .then(data => {
        updateResults(data);
    })
    .catch(error => {
        showError('An error occurred while communicating with the server');
        console.error('Error:', error);
    })
    .finally(() => {
        // Reset button state
        submitButton.disabled = false;
        submitButton.textContent = originalText;
    });
}); 