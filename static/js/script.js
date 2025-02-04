// Basic client-side validation
document.addEventListener('DOMContentLoaded', function() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            const inputs = form.querySelectorAll('input[required]');
            
            inputs.forEach(input => {
                if (!input.value.trim()) {
                    event.preventDefault();
                    input.style.borderColor = 'red';
                    alert('Please fill in all required fields');
                } else {
                    input.style.borderColor = '';
                }
            });
        });
    });
});
