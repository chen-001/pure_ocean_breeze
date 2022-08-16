__updated__ = '2022-08-16 15:21:37'

from tenacity import retry
from loguru import logger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.application import MIMEApplication

class pure_mail(object):
    def __init__(self, host, user, pwd, port=465):
        self.host = host
        self.user = user
        self.pwd = pwd
        self.port = port

    @retry
    def sendemail(self, tolist, subject, body, lastemail_path):
        message = MIMEMultipart()
        message["Form"] = Header(self.user, "utf-8")
        message["To"] = Header(",".join(tolist), "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        message.attach(MIMEText(body, "plain", "utf-8"))

        for path in lastemail_path:
            att1 = MIMEApplication(open(path, "rb").read())
            att1["Content-Type"] = "application/octet-stream"
            att1.add_header(
                "Content-Disposition", "attachment", filename=path.split("/")[-1]
            )
            message.attach(att1)
        try:
            client = smtplib.SMTP_SSL(self.host, self.port)
            login = client.login(self.user, self.pwd)
            if login and login[0] == 235:
                client.sendmail(self.user, tolist, message.as_string())
                logger.success("邮件发送成功")
            else:
                logger.warning("登录失败")
        except Exception as e:
            logger.error(f"发送失败，原因为{e}")