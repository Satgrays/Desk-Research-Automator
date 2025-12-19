"""
Email service with Resend
Includes publication dates in email
"""
import os
import requests
from typing import List, Dict


def send_research_report(to_email: str, query: str, report: str, sources: List[Dict]) -> bool:
    """
    Send research report via email with publication dates
    
    Args:
        to_email: Recipient email
        query: Research query
        report: Generated report
        sources: List of sources with dates
        
    Returns:
        True if email sent successfully
    """
    resend_api_key = os.getenv("RESEND_API_KEY")
    
    if not resend_api_key:
        print("RESEND_API_KEY not configured")
        return False
    
    print(f"Sending report to {to_email}...")
    
    # Generate HTML with publication dates
    sources_html = ""
    for i, src in enumerate(sources[:6], 1):
        pub_date = src.get('published_date', 'Unknown date')
        sources_html += f"""
        <div style="margin-bottom: 15px; padding: 12px; background: #f9f9f9; border-left: 3px solid #667eea; border-radius: 4px;">
            <strong style="color: #667eea;">[{i}]</strong>
            <a href="{src['url']}" style="color: #333; text-decoration: none; font-weight: 600;">
                {src['title']}
            </a>
            <div style="font-size: 12px; color: #999; margin-top: 5px;">
                Published: {pub_date} | Relevance: {src.get('relevance_score', 'N/A')}
            </div>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; border-radius: 10px; text-align: center;">
                <h1 style="color: white; margin: 0; font-size: 28px;">Research Report</h1>
                <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 14px;">
                    Most Recent Academic Papers
                </p>
            </div>
            
            <div style="margin: 30px 0; padding: 20px; background: #f5f7fa; border-radius: 8px;">
                <h2 style="color: #333; margin: 0 0 10px 0; font-size: 18px;">
                    Research Question
                </h2>
                <p style="color: #555; margin: 0; font-size: 16px; line-height: 1.6;">
                    {query}
                </p>
            </div>
            
            <div style="margin: 30px 0;">
                <h2 style="color: #333; margin: 0 0 15px 0; font-size: 20px;">
                    Executive Summary
                </h2>
                <div style="background: white; padding: 25px; border: 1px solid #e0e0e0; border-radius: 8px; line-height: 1.8; color: #444;">
                    {report.replace(chr(10), '<br><br>')}
                </div>
            </div>
            
            <div style="margin: 30px 0;">
                <h2 style="color: #333; margin: 0 0 15px 0; font-size: 20px;">
                    Recent Sources
                </h2>
                {sources_html}
            </div>
            
            <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #e0e0e0; text-align: center;">
                <p style="color: #999; font-size: 12px; margin: 0;">
                    This report was generated automatically using AI and semantic search with Qdrant
                </p>
                <p style="color: #999; font-size: 12px; margin: 10px 0 0 0;">
                    Research Automator 2024
                </p>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    try:
        response = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {resend_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "from": "Research Automator <no-reply@research-automator.com>",
                "to": [to_email],
                "subject": f"Research Report: {query[:60]}...",
                "html": html_content
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Email sent successfully to {to_email}")
            return True
        else:
            print(f"Error sending email: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception sending email: {e}")
        return False