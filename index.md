---
title: Liyuan's Log
---

# Welcome to Liyuan's Log

This is the place to share my learnings!

{% include social-links.html %}

<style>
.post-container {
  margin-top: 30px;
  margin-bottom: 10px;
}

a.post-link {
  display: block;
  text-decoration: none;
  color: inherit;
}

.post-box {
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  padding: 20px;
  background-color: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.post-box:hover {
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.main-content .post-box h2.post-title {
  color: #000000;
  margin-top: 0;
}

.post-date {
  color: #666;
  font-style: normal;
  margin-bottom: 15px;
  font-size: 15px;
}

.post-content {
  color: #666;
  font-size: 15px;
}

/* Override any Cayman theme link colors for the entire box */
.main-content .post-link:hover {
  text-decoration: none;
}

.main-content .post-box h2.post-title a {
  color: #000000;
}
</style>

{% assign sorted_posts = site.pages | where_exp: "item", "item.path contains 'writings/'" | sort: "date" | reverse %}

{% for post in sorted_posts %}
<div class="post-container">
  <a class="post-link" href="{{ post.url | relative_url }}">
    <div class="post-box">
      <h2 class="post-title">{{ post.title }}</h2>
      <div class="post-content">
        {{ post.excerpt }}
      </div>
      <p class="post-date">Date: {{ post.date | date: "%B %-d, %Y" }}</p>
    </div>
  </a>
</div>
{% endfor %}
