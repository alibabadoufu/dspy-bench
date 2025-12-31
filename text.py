def get_card_html(value, trend="up"):
    # Determine colors based on trend (matching the React logic)
    icon_bg = "bg-green-50 text-green-600" if trend == "up" else "bg-blue-50 text-blue-600"
    
    return f"""
    <!-- Tailwind CSS (Required for styling) -->
    <script src="https://cdn.tailwindcss.com"></script>

    <!-- Card Component -->
    <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 flex items-start justify-between font-sans max-w-sm">
        <div>
            <p class="text-sm font-medium text-slate-500 mb-1">Distinct Clients Active</p>
            <h3 class="text-2xl font-bold text-slate-800">{value}</h3>
            <p class="text-xs text-slate-400 mt-1">Total active accounts</p>
        </div>
        
        <!-- Icon Container -->
        <div class="p-3 rounded-lg {icon_bg}">
            <!-- Users Icon (SVG) -->
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M22 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
            </svg>
        </div>
    </div>
    """

# Usage in Gradio:
# with gr.Blocks() as demo:
#     gr.HTML(get_card_html(value="142", trend="up"))

