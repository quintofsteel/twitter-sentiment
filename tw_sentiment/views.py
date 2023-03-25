from django.shortcuts import render


def sentiment_analysis(request):
    return render(request, 'sentiment_analysis.html')

