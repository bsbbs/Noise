from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

url = 'https://mytime.nyumc.org/wfcstatic/applications/navigator/html5/dist/container/index.html?version=8.1.13.223#/'
driver = webdriver.Chrome()  # You need to have the Chrome driver installed
driver.get(url)
username_input = driver.find_element('xpath', '//input[@type="text"]')
password_input = driver.find_element('xpath', '//input[@type="password"]')
submit_button = driver.find_element('xpath', '//input[@type="submit"]')
# Input your values
username_input.send_keys('bs3667')
password_input.send_keys('Zainali20@3')
submit_button.click()


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Define XPath for the input elements with type='hidden'
xpath = '//input[@type="hidden" and @name="factor" and @value="Phone Call"]'
driver.switch_to.frame("duo_iframe")

elements = driver.find_elements('xpath', xpath)
elements[0].click()

# Strategy 3: Using JavaScript to click the element
try:
    element = driver.find_element(By.XPATH, xpath)
    driver.execute_script("arguments[0].click();", element)
    print("Element clicked successfully using JavaScript!")
except:
    print("Failed to click the element using JavaScript.")


# Wait until the input elements are present, and get the list of elements
try:
    elements = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, xpath))
    )
    print(f"Found {len(elements)} hidden input elements.")
    # Iterating through the list of elements
    for i, element in enumerate(elements):
        # Perform actions or print attributes of each element
        if i  == 0:
            print(element.get_attribute('name'), element.get_attribute('value'))
            element.click()
except:
    print("No hidden input elements found!")

# Close the Chrome WebDriver instance
driver.quit()



xpath = "//input[@type='hidden' and @name='factor' and @value='Phone Call']"
xpath = "//input[@name='request_id']"
xpath = "//input[@type='hidden' and @name='url']"
callme = driver.find_element('xpath', xpath)

parent_element = driver.find_element('id', 'duo_iframe')
driver.find_element('action', '/frame/prompt')
callme = parent_element.find_element('xpath', '//input[@type="hidden" & @name="factor"]')

parent_element1 = parent_element.find_element('class', 'auth_methods')

s_element = parent_element.find_element('id', 'auth_methods')
#parent_element = parent_element.find_element('xpath', 'login-form')
#parent_element.find_element('xpath', '//form[@action="/frame/prompt"]')
#parent_element = driver.find_element('xpath', '//form[@action="/frame/prompt"]')
callme = parent_element.find_element('xpath', '//button[@tabindex="2"]')


callme = driver.find_element('xpath', '//input[@type="hidden"]')#@value="Phone Call"]")
callme = driver.find_element('xpath', '//input[@value="Phone Call"]')#@value="Phone Call"]")

value_to_search = 'Value to Search'
xpath_expression = f"//*[contains(text(), '{value_to_search}') or @*[contains(., '{value_to_search}')]]"
# Find the element using the constructed XPath
element = driver.find_element('xpath', xpath_expression)

callme.find_element('id', 'Phone Call')

submit_button = driver.find_element('xpath', '//input[@type="submit"]')

callme = driver.find_element('xpath', '//input[@type="hidden" and @value="Phone Call"]')

phone1 = driver.find_element('xpath', '//fieldset[@data-device-index="phone1"]')

duolist.click()
callme = duolist

callme = parent_element.find_element('xpath', '//input[@value="Phone Call"]')

callme = driver.find_element('xpath', '//input[@type="hidden"]')
callme.click()

driver.quit()


url = 'https://sso.nyumc.org/oam/server/auth_cred_submit'
driver = webdriver.Chrome()  # You need to have the Chrome driver installed
driver.get(url)
clickhere = driver.find_element('xpath', '//a[@href="https://insidehealth.nyumc.org"]')
clickhere.click()

form = driver.find_element('xpath', '//div[@role="main"]')
form = driver.find_element('id', 'auth_methods')




