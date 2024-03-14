__updated__ = "2022-08-18 03:23:01"

from tenacity import retry
from loguru import logger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.application import MIMEApplication


class pure_mail(object):
    def __init__(self, host: str, user: str, pwd: str, port: int = 465) -> None:
        """设置邮箱的账号和授权信息

        Parameters
        ----------
        host : str
            邮箱的服务器地址，如163邮箱的地址为`smtp.163.com`
        user : str
            邮箱账号，形如`xxx@xxx.com`
        pwd : str
            邮箱SMTP服务授权码，在邮箱设置中查看
        port : int, optional
            端口号, by default 465
        """
        self.host = host
        self.user = user
        self.pwd = pwd
        self.port = port

    @retry
    def sendemail(
        self, tolist: list, subject: str, body: str, lastemail_path: list = None
    ) -> None:
        """向指定邮箱账号发送带多个附件的邮件

        Parameters
        ----------
        tolist : list
            发送目标的账号，形如`['xxx@xxx.com','yyy@yyy.com']`
        subject : str
            邮件的主题
        body : str
            邮件的正文
        lastemail_path : list
            附件所在的地址，形如`['/xxx/xxx/xxx.csv','/yyy/yyy/yyy.png']`
        """
        message = MIMEMultipart()
        message["Form"] = Header(self.user, "utf-8")
        message["To"] = Header(",".join(tolist), "utf-8")
        message["Subject"] = Header(subject, "utf-8")
        message.attach(MIMEText(body, "plain", "utf-8"))

        if lastemail_path is not None:
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
