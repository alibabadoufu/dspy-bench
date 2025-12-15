import gradio as gr
import pandas as pd

# 1. Create dummy data
def get_data():
    return pd.DataFrame({
        "Task ID": [101, 102, 103, 104, 105],
        "Feature": ["Login Page", "Dashboard API", "User Settings", "Dark Mode", "Export PDF"],
        "Owner": ["Alice", "Bob", "Charlie", "Alice", "Dave"],
        "Status": ["Done", "In Progress", "Review", "Pending", "Backlog"],
        "Priority": ["High", "Critical", "Medium", "Low", "Low"]
    })

# 2. The JavaScript "Hack"
# This function constructs an HTML table with inline CSS (essential for Teams)
# and writes it to the clipboard as a MIME type 'text/html'.
js_copy_logic = """
async (df_data) => {
    // df_data comes from Gradio as an object: { headers: [...], data: [[...], [...]] }
    if (!df_data || !df_data.data) {
        return "No data found to copy.";
    }

    const headers = df_data.headers;
    const rows = df_data.data;

    // Build HTML string with inline styles. 
    // Teams/Outlook rely on inline CSS for formatting.
    let html = `
    <table style="border-collapse: collapse; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; width: 100%; font-size: 14px;">
        <thead>
            <tr style="background-color: #5b5fc7; color: white;">
    `;

    // Add Headers
    headers.forEach(h => {
        html += `<th style="border: 1px solid #ddd; padding: 10px; text-align: left;">${h}</th>`;
    });

    html += `</tr></thead><tbody>`;

    // Add Rows
    rows.forEach((row, index) => {
        // Alternating row colors for readability
        const bg = index % 2 === 0 ? '#ffffff' : '#f3f2f1';
        html += `<tr style="background-color: ${bg};">`;
        
        row.forEach(cell => {
            html += `<td style="border: 1px solid #d1d1d1; padding: 8px; color: #242424;">${cell}</td>`;
        });
        
        html += `</tr>`;
    });

    html += `</tbody></table>`;

    // Create the ClipboardItems
    // We provide 'text/html' for Teams and 'text/plain' as a fallback for Notepad
    try {
        const blobHtml = new Blob([html], { type: "text/html" });
        const blobText = new Blob([rows.map(r => r.join("\\t")).join("\\n")], { type: "text/plain" });
        
        const data = [new ClipboardItem({
            "text/html": blobHtml,
            "text/plain": blobText
        })];

        await navigator.clipboard.write(data);
        return "✅ Copied to clipboard! Ready to paste into Teams.";
    } catch (err) {
        console.error(err);
        return "❌ Error copying: " + err.message;
    }
}
"""

# 3. Build the Gradio App
with gr.Blocks(title="Teams Copy Hack") as demo:
    gr.Markdown("## 📋 Gradio DataFrame to Teams Copy Hack")
    gr.Markdown("The standard copy button destroys formatting. Use the **Copy for Teams** button below to generate a perfectly formatted table for MS Teams chats.")
    
    with gr.Row():
        # The DataFrame component
        table = gr.DataFrame(value=get_data(), label="Project Status", interactive=True)
    
    with gr.Row():
        # Our custom hack button
        copy_btn = gr.Button("📋 Copy for Teams", variant="primary", scale=0)
        status_msg = gr.Textbox(label="Status", interactive=False, scale=1)

    # 4. Wire the button
    # We pass the 'table' component as an input to the JS function.
    # The JS function handles the clipboard logic.
    # The output is sent to 'status_msg'.
    copy_btn.click(
        fn=None, # Pure JS, no Python function needed
        inputs=[table],
        outputs=[status_msg],
        js=js_copy_logic
    )

if __name__ == "__main__":
    demo.launch()

