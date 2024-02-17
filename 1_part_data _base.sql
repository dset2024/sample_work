
CREATE DATABASE GosGP;


CREATE TABLE TypeOfBudget
(
 id_typeOfBudget  serial NOT NULL ,
 n_TypeOfBudget varchar(100) NOT NULL,
 PRIMARY KEY (id_typeOfBudget)

);

CREATE TABLE Classifier
(
 id_Classifier  serial NOT NULL,
 n_OKPD2        varchar(200) NOT NULL,
 OKPD2          varchar(5) NOT NULL,
 PRIMARY KEY (id_Classifier)

); 

CREATE TABLE StatusContract
(
 id_StatusContract  serial NOT NULL ,
 n_StatusContract varchar(30) NOT NULL,
 PRIMARY KEY (id_StatusContract)

);


CREATE TABLE GovernmentProcurement
(
 id_GovernmentProcurement                 serial NOT NULL,

 n_GovernmentProcurement                text NOT NULL,
 IdentificationCode_GovernmentProcurement varchar(36) NOT NULL,
 inn_Customer    varchar(12) NOT NULL,
 n_Customer      text NOT NULL,
 region_Customer varchar(100) NOT NULL,
 inn_Supplyer varchar(12) NOT NULL,
 n_Supplyer   text NOT NULL,
 number_Contract      varchar(50) NOT NULL,
 signingDate_Contract date NOT NULL,
 startDate_Contract   date NOT NULL,
 endDate_Contract     date NOT NULL,
 cost_Contract        numeric(15,2) NOT null,
  id_Classifier                            integer NOT NULL,
 id_typeOfBudget                          integer NOT NULL,
 id_StatusContract      integer NOT NULL,
 PRIMARY KEY (id_GovernmentProcurement),
 FOREIGN KEY (id_StatusContract) REFERENCES StatusContract (id_StatusContract),
 FOREIGN KEY (id_typeOfBudget) REFERENCES TypeOfBudget (id_typeOfBudget),
 FOREIGN KEY (id_Classifier) REFERENCES Classifier (id_Classifier)
 );
