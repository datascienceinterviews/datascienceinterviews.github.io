var enumBody = document.getElementsByClassName("enumerate-headings-plugin enumerate-heading-plugin");
for (var i = 0; i < enumBody.length; i++) {
	enumBody[i].innerHTML = enumBody[i].innerHTML.replace("1.", "");
}
var enumNavi = document.getElementsByClassName("md-nav__link")
for (var i = 0; i < enumNavi.length; i++) {
	enumNavi[i].innerHTML = enumNavi[i].innerHTML.replace("1.", "");
}

// Make Free Tools link open in a new tab
document.addEventListener('DOMContentLoaded', function () {
	var links = document.querySelectorAll('a[href="https://onlinetoolsvault.com"]');
	links.forEach(function (link) {
		link.setAttribute('target', '_blank');
		link.setAttribute('rel', 'noopener noreferrer');
	});
});