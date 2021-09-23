from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

driver = webdriver.Firefox(executable_path="C:/driver/geckodriver.exe")
driver.wait = WebDriverWait(driver, 2)
driver.get('https://search.naver.com/search.naver?sm=tab_hty.top&where=image&query=%EB%94%B8%EA%B8%B0')
image_elements = driver.find_elements_by_xpath("/img")
for image in image_elements:
    img_src = image.get_attribute("src")
    alt = image.get_attribute("alt")

print(alt)