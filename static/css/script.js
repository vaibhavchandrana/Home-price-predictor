var sliderimg=document.getElementById("sliderimg");
var images=new Array("../static/images/background1.jpg","../static/images/background2.jpg",
                        "../static/images/background3.jpg",
                            "../static/images/background4.jpg","../static/images/background5.jpg");
var len=images.length;
var i=0;
function slider()
{
    if(i>len-1)
    {
        i=0;
    }
    sliderimg.src=images[i];
    i++;
    setTimeout('slider()',3000);
}