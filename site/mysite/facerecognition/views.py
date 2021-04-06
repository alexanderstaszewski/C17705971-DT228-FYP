from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from facerecognition.tools import getFaces
import os
from pathlib import Path
from django.conf import settings
from django.contrib import messages
from .forms import UploadForm

mediaPath = settings.MEDIA_ROOT
staticPath = settings.STATICFILES_DIRS[0]

def home_view(request, *args, **kwargs):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            classifyCharacters = True
            filename, facePath, outputPath = getPaths(request.FILES['file'])

            if len(request.POST.getlist('ignore_character_identification')) != 0:
                classifyCharacters = False
            
            facesCount, charactersFound = getFaces(facePath, outputPath, classifyCharacters)
            
            print(charactersFound)
            
            return render(request, 'result.html', {
                'result': facesCount,
                'characters': charactersFound,
                'facepath' : filename
            })
        
    else:
        form = UploadForm()

    
    return render(request, "home.html", {
        'form' : form
    })

def info_view(request, *args, **kwargs):

    return render(request, "info.html", *args)
    
def about_view(request, *args, **kwargs):

    return render(request, "about.html", *args)
    
def test_view(request, *args, **kwargs):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        #uploadedFileUrl = fs.url(filename)
        facePath = mediaPath + "\\" + filename
        
        #prediction = isFace(facePath)
        outputPath = staticPath + "\\media\\" + filename
        facesCount, charactersFound = getFaces(facePath, outputPath)
        print(facesCount)

        print(filename)
        
        messages.info(request, "Error Message")
        
        
        return render(request, 'result.html', {
            'result': facesCount,
            'characters': charactersFound,
            'facepath': filename
        })
    return render(request, "test.html", {})
    
def getPaths(file):
    fs = FileSystemStorage()
    filename = fs.save(file.name, file)
    facePath = mediaPath + "\\" + filename
    outputPath = staticPath + "\\media\\" + filename
    
    return filename, facePath, outputPath
